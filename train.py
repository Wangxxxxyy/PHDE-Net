import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from collections import OrderedDict
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import Resize
import arch as archs
from dataset import Dataset
from metrics import mIoU, nIoU, PD_FA, F1Score
from utils import AverageMeter, str2bool
from tensorboardX import SummaryWriter

class BCEDiceLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        dims = tuple(range(1, prob.ndim))
        intersection = (prob * target).sum(dim=dims)
        denominator = prob.sum(dim=dims) + target.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return bce + (1.0 - dice).mean()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('-b', '--batch_size', default=8, type=int)
    parser.add_argument('--arch', '-a', default='PHDE_Net')
    parser.add_argument('--input_channels', default=3, type=int)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--input_w', default=256, type=int)
    parser.add_argument('--input_h', default=256, type=int)
    parser.add_argument('--loss', default='BCEDiceLoss', choices=['BCEDiceLoss', 'BCEWithLogitsLoss'])
    parser.add_argument('--rec_lambda', default=0.15, type=float)
    parser.add_argument('--bg_supp_lambda', default=0.08, type=float)
    parser.add_argument('--dataset', default='IRSTD-1K')
    parser.add_argument('--data_dir', default='./')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--nesterov', default=False, type=str2bool)
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float)
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--early_stopping', default=-1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--val_ratio', default=0.1, type=float,
                        help='内部验证集比例；默认0.1，从train.txt固定划分10%%作为验证集')
    parser.add_argument('--split_seed', default=1029, type=int)
    return parser.parse_args()

def load_split_ids(base_path, val_ratio=0.0, seed=1029):
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError('val_ratio 必须在 [0, 1) 范围内')

    with open(os.path.join(base_path, 'img_idx', 'train.txt'), 'r', encoding='utf-8') as f:
        all_train_ids = [line.strip() for line in f if line.strip()]

    if not all_train_ids:
        raise ValueError('train.txt 为空，请检查数据集划分文件')
    if val_ratio == 0.0:
        return all_train_ids, []
    if len(all_train_ids) < 2:
        raise ValueError('启用内部验证划分时，train.txt 至少需要 2 个样本')

    shuffled_ids = all_train_ids.copy()
    random.Random(seed).shuffle(shuffled_ids)
    val_count = max(1, int(len(shuffled_ids) * val_ratio))
    val_count = min(val_count, len(shuffled_ids) - 1)
    return shuffled_ids[val_count:], shuffled_ids[:val_count]

REC_WEIGHTS = [0.05, 0.15, 0.4, 1.0]

def masked_asymmetric_rec_loss(recon, target, mask):
    diff = recon - target
    sq_loss = diff ** 2
    bg_mask = 1.0 - mask
    masked_loss = sq_loss * bg_mask
    return torch.where(diff > 0, 2.0 * masked_loss, masked_loss).mean()

def compute_rec_loss(recons, x_input, masks):
    loss_rec = 0.0
    for rec, w in zip(recons, REC_WEIGHTS):
        size_i = rec.shape[2:]
        target_i = F.interpolate(x_input, size=size_i, mode='bilinear', align_corners=False)
        mask_i = F.interpolate(masks, size=size_i, mode='nearest')
        loss_rec = loss_rec + w * masked_asymmetric_rec_loss(rec, target_i, mask_i)
    return loss_rec

def compute_bg_suppression_loss(bg_gens, masks):
    loss_supp = 0.0
    supp_weights = [0.05, 0.15, 0.4, 1.0]
    for bg, w in zip(bg_gens, supp_weights):
        size_i = bg.shape[2:]
        mask_i = F.interpolate(masks, size=size_i, mode='nearest')
        loss_supp = loss_supp + w * (bg.abs() * mask_i).mean()
    return loss_supp

def train(config, train_loader, model, criterion, optimizer, epoch):
    avg_meters = {'loss': AverageMeter(), 'seg_loss': AverageMeter(),
                  'rec_loss': AverageMeter(), 'supp_loss': AverageMeter()}
    miou_metric = mIoU()
    model.train()
    pbar = tqdm(total=len(train_loader), desc=f'Train [{epoch}]')
    device = next(model.parameters()).device
    for input, target, _ in train_loader:
        input, target = input.to(device, non_blocking=True), target.to(device, non_blocking=True)
        seg_out, recons, bg_gens = model(input)
        loss_seg = criterion(seg_out, target)
        loss_rec = compute_rec_loss(recons, input, target)
        loss_supp = compute_bg_suppression_loss(bg_gens, target)
        loss = loss_seg + config['rec_lambda'] * loss_rec + config['bg_supp_lambda'] * loss_supp
        pred = (torch.sigmoid(seg_out) > 0.5).float()
        miou_metric.update(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bs = input.size(0)
        avg_meters['loss'].update(loss.item(), bs)
        avg_meters['seg_loss'].update(loss_seg.item(), bs)
        avg_meters['rec_loss'].update(loss_rec.item(), bs)
        avg_meters['supp_loss'].update(loss_supp.item(), bs)
        pbar.set_postfix(OrderedDict([
            ('loss', f"{avg_meters['loss'].avg:.4f}"),
            ('seg', f"{avg_meters['seg_loss'].avg:.4f}"),
            ('rec', f"{avg_meters['rec_loss'].avg:.4f}"),
            ('supp', f"{avg_meters['supp_loss'].avg:.4f}"),
            ('alpha1', f"{torch.sigmoid(model.alpha1).item():.2f}"),
        ]))
        pbar.update(1)
    pbar.close()
    pixAcc, mIoU_value = miou_metric.get()
    alphas = {f'a{i}': torch.sigmoid(getattr(model, f'alpha{i}')).item() for i in range(1, 5)}
    res = OrderedDict([('loss', avg_meters['loss'].avg), ('seg_loss', avg_meters['seg_loss'].avg),
                       ('rec_loss', avg_meters['rec_loss'].avg), ('supp_loss', avg_meters['supp_loss'].avg),
                       ('pixAcc', pixAcc), ('mIoU', mIoU_value)])
    res.update(alphas)
    return res

def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(), 'seg_loss': AverageMeter(),
                  'rec_loss': AverageMeter(), 'supp_loss': AverageMeter()}
    miou_metric = mIoU()
    niou_metric = nIoU(nclass=1, score_thresh=0.5)
    pd_fa_metric = PD_FA()
    f1_metric = F1Score(nclass=1, bins=10)
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader), desc='Val')
        for input, target, _ in val_loader:
            input, target = input.to(device, non_blocking=True), target.to(device, non_blocking=True)
            seg_out, recons, bg_gens = model(input)
            loss_seg = criterion(seg_out, target)
            loss_rec = compute_rec_loss(recons, input, target)
            loss_supp = compute_bg_suppression_loss(bg_gens, target)
            loss = loss_seg + config['rec_lambda'] * loss_rec + config['bg_supp_lambda'] * loss_supp
            pred = (torch.sigmoid(seg_out) > 0.5).float()
            miou_metric.update(pred, target)
            niou_metric.update(torch.sigmoid(seg_out), target)
            f1_metric.update(torch.sigmoid(seg_out), target)
            for i in range(pred.size(0)):
                pd_fa_metric.update(pred[i].squeeze(), target[i].squeeze(),
                                    (config['input_h'], config['input_w']))
            bs = input.size(0)
            avg_meters['loss'].update(loss.item(), bs)
            avg_meters['seg_loss'].update(loss_seg.item(), bs)
            avg_meters['rec_loss'].update(loss_rec.item(), bs)
            avg_meters['supp_loss'].update(loss_supp.item(), bs)
            pbar.set_postfix(OrderedDict([('loss', f"{avg_meters['loss'].avg:.4f}"),
                                          ('seg', f"{avg_meters['seg_loss'].avg:.4f}")]))
            pbar.update(1)
        pbar.close()
    pixAcc, mIoU_value = miou_metric.get()
    nIoU_value = niou_metric.get()
    PD, FA = pd_fa_metric.get()
    F1 = f1_metric.get()
    return OrderedDict([('loss', avg_meters['loss'].avg), ('seg_loss', avg_meters['seg_loss'].avg),
                        ('rec_loss', avg_meters['rec_loss'].avg), ('supp_loss', avg_meters['supp_loss'].avg),
                        ('pixAcc', pixAcc), ('mIoU', mIoU_value), ('nIoU', nIoU_value),
                        ('PD', PD), ('FA', FA), ('F1', F1)])

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def main():
    seed_torch()
    config = vars(parse_args())
    if config['name'] is None:
        config['name'] = '%s_%s' % (config['dataset'], config['arch'])
    exp_name = config['name']
    output_dir = config['output_dir']
    experiment_dir = os.path.join(output_dir, exp_name)
    os.makedirs(experiment_dir, exist_ok=True)
    my_writer = SummaryWriter(experiment_dir)

    print('=' * 80)
    print('PHDE-Net (paper-aligned implementation)')
    print('  Encoder channels: C1=32, C2=64, C3=128, C4=128')
    print('  SSC: 3x7 / 7x3 / 5x5 / Identity + GAP/GMP + Softmax')
    print('  THP: bilinear upsample + depthwise-separable structure path + modulation + AvgPool')
    print('  AHS: learnable subtraction + gated negative residual + diff-guided refinement')
    print('  Loss: BCE+Dice + 0.15*L_recon + 0.08*L_bg')
    print(f'  rec_lambda={config["rec_lambda"]}  bg_supp_lambda={config["bg_supp_lambda"]}')
    print(f'  val_ratio={config["val_ratio"]} (default 0.1 reserves 10% of train.txt for validation)')
    print('=' * 80)

    with open(os.path.join(experiment_dir, 'config.yml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, allow_unicode=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = (
        nn.BCEWithLogitsLoss()
        if config['loss'] == 'BCEWithLogitsLoss'
        else BCEDiceLoss()
    ).to(device)
    model = archs.__dict__[config['arch']](
        num_classes=config['num_classes'], input_channels=config['input_channels']
    ).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    print(f'  Device: {device}; total params: {total_p / 1e6:.2f}M')

    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']
        )
    else:
        optimizer = optim.SGD(
            model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'],
            momentum=config['momentum'], nesterov=config['nesterov']
        )

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr']
        )
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=config['factor'], patience=config['patience'],
            min_lr=config['min_lr']
        )
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(e) for e in config['milestones'].split(',')],
            gamma=config['factor']
        )
    else:
        scheduler = None

    base_path = os.path.join(config['data_dir'], config['dataset'])
    train_ids, val_ids = load_split_ids(
        base_path, val_ratio=config['val_ratio'], seed=config['split_seed']
    )
    print(f'  Train samples: {len(train_ids)}; validation samples: {len(val_ids)}')
    with open(os.path.join(experiment_dir, 'train_ids.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_ids) + '\n')
    with open(os.path.join(experiment_dir, 'val_ids.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_ids) + ('\n' if val_ids else ''))

    
    train_transform = Compose([
        Resize(config['input_h'], config['input_w']), transforms.Normalize()
    ])
    train_dataset = Dataset(
        img_ids=train_ids,
        img_dir=os.path.join(base_path, 'images'),
        mask_dir=os.path.join(base_path, 'masks'),
        img_ext='.png', mask_ext='.png', num_classes=config['num_classes'],
        transform=train_transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], drop_last=False
    )

    val_loader = None
    if val_ids:
        val_dataset = Dataset(
            img_ids=val_ids,
            img_dir=os.path.join(base_path, 'images'),
            mask_dir=os.path.join(base_path, 'masks'),
            img_ext='.png', mask_ext='.png', num_classes=config['num_classes'],
            transform=train_transform
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config['batch_size'], shuffle=False,
            num_workers=config['num_workers'], drop_last=False
        )

    log = OrderedDict([
        ('epoch', []), ('lr', []), ('loss', []), ('seg_loss', []),
        ('rec_loss', []), ('supp_loss', []), ('pixAcc', []), ('mIoU', []),
        ('val_loss', []), ('val_seg_loss', []), ('val_rec_loss', []), ('val_supp_loss', []),
        ('val_pixAcc', []), ('val_mIoU', []), ('val_nIoU', []),
        ('val_PD', []), ('val_FA', []), ('val_F1', []),
        ('a1', []), ('a2', []), ('a3', []), ('a4', [])
    ])
    best_miou = -1.0
    trigger = 0
    checkpoint_path = os.path.join(experiment_dir, 'model.pth')
    print('\n' + '=' * 80 + '\nStart Training\n' + '=' * 80)

    for epoch in range(config['epochs']):
        current_lr = optimizer.param_groups[0]['lr']
        print(f'\nEpoch [{epoch + 1}/{config["epochs"]}]  LR: {current_lr:.2e}')
        train_log = train(config, train_loader, model, criterion, optimizer, epoch)
        val_log = validate(config, val_loader, model, criterion) if val_loader is not None else None

        if scheduler is not None:
            if config['scheduler'] == 'ReduceLROnPlateau':
                monitored_loss = val_log['loss'] if val_log is not None else train_log['loss']
                scheduler.step(monitored_loss)
            else:
                scheduler.step()

        if val_log is not None:
            print(
                'loss %.4f (seg %.4f rec %.4f bg %.4f) train_mIoU %.4f | '
                'val_mIoU %.4f nIoU %.4f Pd %.4f Fa %.2e F1 %.4f' % (
                    train_log['loss'], train_log['seg_loss'], train_log['rec_loss'],
                    train_log['supp_loss'], train_log['mIoU'], val_log['mIoU'],
                    val_log['nIoU'], val_log['PD'], val_log['FA'], val_log['F1']
                )
            )
            if val_log['mIoU'] > best_miou:
                torch.save(model.state_dict(), checkpoint_path)
                best_miou = val_log['mIoU']
                trigger = 0
                print('=> saved best validation mIoU checkpoint')
            else:
                trigger += 1
        else:
            print(
                'loss %.4f (seg %.4f rec %.4f bg %.4f) train_mIoU %.4f' % (
                    train_log['loss'], train_log['seg_loss'], train_log['rec_loss'],
                    train_log['supp_loss'], train_log['mIoU']
                )
            )
            
            torch.save(model.state_dict(), checkpoint_path)

        val_keys = {
            'loss': 'loss', 'seg_loss': 'seg_loss', 'rec_loss': 'rec_loss',
            'supp_loss': 'supp_loss', 'pixAcc': 'pixAcc', 'mIoU': 'mIoU',
            'nIoU': 'nIoU', 'PD': 'PD', 'FA': 'FA', 'F1': 'F1'
        }
        log['epoch'].append(epoch + 1)
        log['lr'].append(current_lr)
        for key in ('loss', 'seg_loss', 'rec_loss', 'supp_loss', 'pixAcc', 'mIoU'):
            log[key].append(train_log[key])
        for log_key, val_key in val_keys.items():
            log['val_' + log_key].append(val_log[val_key] if val_log is not None else float('nan'))
        for i in range(1, 5):
            log[f'a{i}'].append(train_log[f'a{i}'])
        pd.DataFrame(log).to_csv(os.path.join(experiment_dir, 'log.csv'), index=False)

        for key in ('loss', 'seg_loss', 'rec_loss', 'supp_loss', 'mIoU'):
            my_writer.add_scalar(f'train/{key}', train_log[key], global_step=epoch)
        my_writer.add_scalar('train/lr', current_lr, global_step=epoch)
        for i in range(1, 5):
            my_writer.add_scalar(f'train/alpha{i}', train_log[f'a{i}'], global_step=epoch)
        if val_log is not None:
            for key in ('loss', 'mIoU', 'nIoU', 'PD', 'FA', 'F1', 'rec_loss', 'supp_loss'):
                my_writer.add_scalar(f'val/{key}', val_log[key], global_step=epoch)

        if val_loader is not None and config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print('=> early stopping')
            break
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if val_loader is not None:
        print(f'\nTraining finished! Best validation mIoU: {best_miou:.4f}')
    else:
        print(f'\nTraining finished! Final-epoch checkpoint saved to: {checkpoint_path}')
    my_writer.close()

if __name__ == '__main__':
    main()
