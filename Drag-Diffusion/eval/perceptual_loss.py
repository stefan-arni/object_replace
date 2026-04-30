import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF


class VGGPerceptualLoss(nn.Module):
    """
    VGG-16 relu3_3 perceptual distance as an LPIPS proxy.
    Lower score = better texture preservation.
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, device: torch.device):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features = vgg.features[:16].to(device=device, dtype=torch.float32)
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.device = device

    def _preprocess(self, t: torch.Tensor) -> torch.Tensor:
        """[1,3,H,W] in [-1,1] → ImageNet-normalized."""
        t = (t.float().clamp(-1, 1) + 1.0) / 2.0  # [0,1]
        mean = torch.tensor(self.IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(self.IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        return (t - mean) / std

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Returns scalar MSE between relu3_3 feature maps."""
        f1 = self.features(self._preprocess(img1))
        f2 = self.features(self._preprocess(img2))
        return torch.mean((f1 - f2) ** 2).item()
