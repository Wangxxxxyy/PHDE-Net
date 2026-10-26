
import torch
from torch import nn
import torch.nn.functional as F

__all__ = ["PHDE_Net"]


class SSCMLP(nn.Module):

    def __init__(self, channels: int, hidden_channels: int | None = None):
        super().__init__()
        hidden_channels = hidden_channels or max(channels // 4, 1)
        self.fc1 = nn.Conv2d(2 * channels, hidden_channels, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_channels, 4 * channels, kernel_size=1)

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(descriptor)))


class SSC(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.horizontal = nn.Conv2d(
            channels, channels, kernel_size=(3, 7), padding=(1, 3), bias=True
        )
        self.vertical = nn.Conv2d(
            channels, channels, kernel_size=(7, 3), padding=(3, 1), bias=True
        )
        self.isotropic = nn.Conv2d(
            channels, channels, kernel_size=5, padding=2, bias=True
        )
        self.weight_mlp = SSCMLP(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        x_h = self.horizontal(x)
        x_w = self.vertical(x)
        x_sym = self.isotropic(x)

        
        scene = x_h + x_w + x_sym + x
        avg_desc = F.adaptive_avg_pool2d(scene, output_size=1)
        max_desc = F.adaptive_max_pool2d(scene, output_size=1)
        descriptor = torch.cat((avg_desc, max_desc), dim=1)

        
        logits = self.weight_mlp(descriptor).reshape(batch, 4, channels, 1, 1)
        weights = torch.softmax(logits, dim=1)

        
        return (
            weights[:, 0] * x_h
            + weights[:, 1] * x_w
            + weights[:, 2] * x_sym
            + weights[:, 3] * x
        )


class ResBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.main(x) + self.shortcut(x), inplace=True)


class THPBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.structure = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, padding=1,
                groups=in_channels, bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.modulation = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )
        
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, previous_ref: torch.Tensor, target_size) -> torch.Tensor:
        up = F.interpolate(
            previous_ref, size=target_size, mode="bilinear", align_corners=False
        )
        generated = self.structure(up)
        modulation = self.modulation(up)
        return self.avg_pool(generated * modulation)


class DiffGuidedRefinement(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, positive_residual: torch.Tensor, diff_abs: torch.Tensor) -> torch.Tensor:
        refined = self.conv1(positive_residual)
        refined = F.relu(self.conv2(refined) + positive_residual, inplace=True)
        beta = self.spatial_attention(diff_abs)
        return beta * refined


class PHDE_Net(nn.Module):

    def __init__(self, num_classes: int = 1, input_channels: int = 3):
        super().__init__()
        
        self.c1, self.c2, self.c3, self.c4 = 32, 64, 128, 128
        c1, c2, c3, c4 = self.c1, self.c2, self.c3, self.c4

        
        self.encoder1 = ResBlock(input_channels, c1)
        self.encoder2 = ResBlock(c1, c2)
        self.encoder3 = ResBlock(c2, c3)
        self.encoder4 = ResBlock(c3, c4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ssc = SSC(c4)

        
        self.thp4 = THPBlock(c4, c4)
        self.thp3 = THPBlock(c4, c3)
        self.thp2 = THPBlock(c3, c2)
        self.thp1 = THPBlock(c2, c1)

        
        
        for i in range(1, 5):
            setattr(self, f"alpha{i}", nn.Parameter(torch.zeros(1)))
            setattr(self, f"gate{i}", nn.Parameter(torch.zeros(1)))

        self.refine4 = DiffGuidedRefinement(c4)
        self.refine3 = DiffGuidedRefinement(c3)
        self.refine2 = DiffGuidedRefinement(c2)
        self.refine1 = DiffGuidedRefinement(c1)

        
        self.recon4 = nn.Conv2d(c4, input_channels, kernel_size=1)
        self.recon3 = nn.Conv2d(c3, input_channels, kernel_size=1)
        self.recon2 = nn.Conv2d(c2, input_channels, kernel_size=1)
        self.recon1 = nn.Conv2d(c1, input_channels, kernel_size=1)

        
        
        
        self.dec_res4 = ResBlock(c4 + c4, c4)
        self.dec_up3 = self._conv3x3(c4, c3)
        self.dec_res3 = ResBlock(c3 + c3, c3)
        self.dec_up2 = self._conv3x3(c3, c2)
        self.dec_res2 = ResBlock(c2 + c2, c2)
        self.dec_up1 = self._conv3x3(c2, c1)
        self.dec_res1 = ResBlock(c1 + c1, c1)
        self.final = nn.Conv2d(c1, num_classes, kernel_size=1)

    @staticmethod
    def _conv3x3(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def ahs_step(
        self,
        encoder_feature: torch.Tensor,
        reference: torch.Tensor,
        alpha: torch.Tensor,
        gate: torch.Tensor,
        refinement: nn.Module,
        recon_projection: nn.Module,
    ):
        
        diff = encoder_feature - torch.sigmoid(alpha) * reference
        positive_residual = F.relu(diff)

        
        negative_compensation = torch.sigmoid(gate) * F.relu(-diff)

        
        foreground = refinement(positive_residual, diff.abs()) + negative_compensation
        reconstruction = recon_projection(reference)
        return foreground, reconstruction

    def forward(self, x: torch.Tensor):
        
        
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        
        
        b = self.ssc(e4)

        
        b4 = self.thp4(b, e4.shape[2:])
        f4, r4 = self.ahs_step(e4, b4, self.alpha4, self.gate4, self.refine4, self.recon4)

        b3 = self.thp3(b4, e3.shape[2:])
        f3, r3 = self.ahs_step(e3, b3, self.alpha3, self.gate3, self.refine3, self.recon3)

        b2 = self.thp2(b3, e2.shape[2:])
        f2, r2 = self.ahs_step(e2, b2, self.alpha2, self.gate2, self.refine2, self.recon2)

        b1 = self.thp1(b2, e1.shape[2:])
        f1, r1 = self.ahs_step(e1, b1, self.alpha1, self.gate1, self.refine1, self.recon1)

        
        
        d4 = self.dec_res4(torch.cat((b, f4), dim=1))

        d3_up = F.interpolate(d4, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3_up = self.dec_up3(d3_up)
        d3 = self.dec_res3(torch.cat((d3_up, f3), dim=1))

        d2_up = F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2_up = self.dec_up2(d2_up)
        d2 = self.dec_res2(torch.cat((d2_up, f2), dim=1))

        d1_up = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1_up = self.dec_up1(d1_up)
        d1 = self.dec_res1(torch.cat((d1_up, f1), dim=1))

        seg_out = self.final(d1)
        if seg_out.shape[2:] != x.shape[2:]:
            seg_out = F.interpolate(
                seg_out, size=x.shape[2:], mode="bilinear", align_corners=False
            )

        
        return seg_out, [r4, r3, r2, r1], [b4, b3, b2, b1]
