import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm
from albumentations import Resize
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose

import arch as archs
from dataset import Dataset
from metrics import mIoU, nIoU, PD_FA, F1Score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='PHDE_Net')
    parser.add_argument('--input_channels', default=3, type=int)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--input_w', default=256, type=int)
    parser.add_argument('--input_h', default=256, type=int)
    parser.add_argument('--dataset', default='IRSTD-1K')
    parser.add_argument('--data_dir', default='./')
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--erode', default=False, action='store_true',
                        help='推理时腐蚀后处理，收缩预测边界1px')
    return parser.parse_args()


def load_split_ids(base_path):
    with open(os.path.join(base_path, 'img_idx', 'test.txt'), 'r') as f:
        return [l.strip() for l in f if l.strip()]


def make_sample_vis(gt_np, pred_np, sample_id,
                    base_path, img_ext='.png'):
    H, W = gt_np.shape

    
    img_path = os.path.join(base_path, 'images', sample_id + img_ext)
    orig_bgr = cv2.imread(img_path)
    if orig_bgr is None:
        
        orig_bgr = cv2.imread(img_path.replace('.png', '.bmp'))
    if orig_bgr is not None:
        orig_bgr = cv2.resize(orig_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        orig_bgr = np.zeros((H, W, 3), dtype=np.uint8)
    vis = orig_bgr

    
    gt_color = np.zeros((H, W, 3), dtype=np.uint8)
    gt_color[gt_np > 0] = [0, 255, 0]

    
    pred_color = np.zeros((H, W, 3), dtype=np.uint8)
    pred_color[pred_np > 0] = [0, 0, 255]

    
    row = np.hstack([vis, gt_color, pred_color])

    
    titles = ['Image', 'GT (green)', 'Pred (red)']
    col_w = W
    label_img = np.full((25, row.shape[1], 3), 200, dtype=np.uint8)
    for i, title in enumerate(titles):
        x = i * col_w + 5
        cv2.putText(label_img, title, (x, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
    row = np.vstack([label_img, row])

    return row


def main():
    config = vars(parse_args())

    if config['model_path'] is None:
        exp_name = '%s_%s' % (config['dataset'], config['arch'])
        config['model_path'] = os.path.join(config['output_dir'], exp_name, 'model.pth')

    print('=' * 80)
    print(f'Model  : {config["model_path"]}')
    print(f'Dataset: {config["dataset"]}')
    print(f'Erode  : {config["erode"]}')
    print('=' * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = archs.__dict__[config['arch']](
        num_classes=config['num_classes'],
        input_channels=config['input_channels']
    ).to(device)
    state_dict = torch.load(config['model_path'], map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f'Device : {device}')
    print(f'Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

    base_path = os.path.join(config['data_dir'], config['dataset'])
    val_ids = load_split_ids(base_path)
    print(f'Test samples: {len(val_ids)}')

    val_transform = Compose([Resize(config['input_h'], config['input_w']), transforms.Normalize()])
    val_dataset = Dataset(
        img_ids=val_ids,
        img_dir=os.path.join(base_path, 'images'),
        mask_dir=os.path.join(base_path, 'masks'),
        img_ext='.png', mask_ext='.png',
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1,
        shuffle=False, num_workers=config['num_workers'], drop_last=False)

    exp_name = '%s_%s' % (config['dataset'], config['arch'])
    out_dir = os.path.join(config['output_dir'], exp_name)
    suffix = '_erode' if config['erode'] else ''
    vis_dir = os.path.join(out_dir, f'visualizations{suffix}')
    os.makedirs(vis_dir, exist_ok=True)
    
    for f in os.listdir(vis_dir):
        if f.endswith('.png'):
            os.remove(os.path.join(vis_dir, f))

    miou_metric = mIoU()
    niou_metric = nIoU(nclass=1, score_thresh=config['threshold'])
    pd_fa_metric = PD_FA()
    f1_metric = F1Score(nclass=1, bins=10)
    per_sample_records = []

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader), desc='Test')
        for idx, (input, target, img_id) in enumerate(val_loader):
            input, target = input.to(device, non_blocking=True), target.to(device, non_blocking=True)
            seg_out, _, _ = model(input)
            prob = torch.sigmoid(seg_out)
            if config['erode']:
                prob = -F.max_pool2d(-prob, kernel_size=3, stride=1, padding=1)
            pred = (prob > config['threshold']).float()

            miou_metric.update(pred, target)
            niou_metric.update(prob, target)
            f1_metric.update(prob, target)
            for i in range(pred.size(0)):
                pd_fa_metric.update(pred[i].squeeze(), target[i].squeeze(),
                                    (config['input_h'], config['input_w']))

            pred_np = pred[0, 0].cpu().numpy().astype(np.int64)
            gt_np   = target[0, 0].cpu().numpy().astype(np.int64)

            intersection = int((pred_np * gt_np).sum())
            pred_area    = int(pred_np.sum())
            gt_area      = int(gt_np.sum())
            union        = pred_area + gt_area - intersection
            sample_iou   = float(intersection) / float(union + 1e-8) if union > 0 else 1.0
            area_ratio   = float(pred_area) / float(gt_area) if gt_area > 0 else float('nan')
            try:
                if isinstance(img_id, dict):
                    sample_id = img_id['img_id']
                elif isinstance(img_id, (list, tuple)):
                    sample_id = img_id[0]['img_id']
                else:
                    sample_id = str(img_id)
            except Exception:
                sample_id = str(idx)
            if isinstance(sample_id, list):
                sample_id = str(sample_id[0])
            per_sample_records.append({
                'img_id':       sample_id,
                'iou':          round(sample_iou, 4),
                'gt_area_px':   gt_area,
                'pred_area_px': pred_area,
                'area_ratio':   round(area_ratio, 4) if not np.isnan(area_ratio) else None,
            })

            
            ratio_int = int(round(area_ratio * 100)) if not np.isnan(area_ratio) else -1
            filename = (f"{sample_id}_iou{int(sample_iou*10000):04d}_"
                        f"ratio{ratio_int:03d}.png")
            vis_img = make_sample_vis(
                gt_np, pred_np,
                sample_id=sample_id, base_path=base_path)
            cv2.imwrite(os.path.join(vis_dir, filename), vis_img)

            
            pred_u8 = (np.clip(pred_np.astype(np.float32), 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(vis_dir, f'{sample_id}_pred_binary.png'), pred_u8)

            pbar.update(1)

    pixAcc, mIoU_val = miou_metric.get()
    nIoU_val = niou_metric.get()
    PD, FA   = pd_fa_metric.get()
    F1       = f1_metric.get()

    print('\n' + '=' * 80)
    print(f'  mIoU  : {mIoU_val:.4f}')
    print(f'  nIoU  : {nIoU_val:.4f}')
    print(f'  PD    : {PD:.4f}')
    print(f'  FA    : {FA:.2e}')
    print(f'  F1    : {F1:.4f}')
    print(f'  pixAcc: {pixAcc:.4f}')
    print('=' * 80)

    df = pd.DataFrame(per_sample_records).sort_values('iou')
    csv_path = os.path.join(out_dir, f'per_sample_iou{suffix}.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nPer-sample IoU saved to: {csv_path}')

    df_valid = df[df['gt_area_px'] > 0].copy()
    df_valid['area_ratio'] = df_valid['area_ratio'].astype(float)
    print('\n--- IoU & area_ratio by target size ---')
    print(f'  {"size range":>12}  {"samples":>7}  {"mean_IoU":>8}  {"mean_ratio":>10}')
    for lo, hi in [(0, 9), (10, 25), (26, 81), (82, 99999)]:
        sub = df_valid[(df_valid['gt_area_px'] >= lo) & (df_valid['gt_area_px'] <= hi)]
        if len(sub) == 0:
            continue
        hi_str = str(hi) if hi < 99999 else '∞'
        print(f'  {lo:5d}~{hi_str:>6} px  {len(sub):>7d}  {sub["iou"].mean():>8.4f}  {sub["area_ratio"].mean():>10.4f}')

    print(f'\n  全局 mean pred/gt ratio: {df_valid["area_ratio"].mean():.4f}')
    print(f'  虚胖(ratio>1): {(df_valid["area_ratio"] > 1).mean()*100:.1f}%')
    print(f'  过减(ratio<1): {(df_valid["area_ratio"] < 1).mean()*100:.1f}%')


if __name__ == '__main__':
    main()
