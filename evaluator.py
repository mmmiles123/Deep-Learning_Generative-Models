import torch
import torch.nn as nn
import torchvision.models as models

class Evaluator(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()
        # ✅ 載入 ImageNet 預訓練的 ResNet18
        self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits = self.resnet18(x)
        return self.sigmoid(logits)

    def eval(self, imgs, labels, threshold=0.5):
        """
        imgs: shape [N, 3, 64, 64]
        labels: shape [N, num_classes] multi-hot vectors
        """
        self.resnet18.eval()
        with torch.no_grad():
            preds = self(imgs)
            preds_bin = (preds > threshold).float()

            # 計算準確率：所有 class 的平均
            correct = (preds_bin == labels).float()
            acc_per_sample = correct.mean(dim=1)  # 每筆資料的平均
            acc = acc_per_sample.mean()           # 所有樣本平均

        return acc.item()
