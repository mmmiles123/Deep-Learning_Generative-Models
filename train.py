import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from model import ConditionalUNetDiffusers
from ddpm import DDPM
from dataset import ICLEVRDataset, load_obj2idx
from evaluator import Evaluator

# 設定參數
epochs = 200
batch_size = 64
lr = 1e-4
save_path = "checkpoints/best_ddpm.pt"

# 裝置選擇
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用裝置：{device}")
# 載入物件對應字典
obj2idx, idx2obj = load_obj2idx("objects.json")
num_classes = len(obj2idx)

# 建立資料集與 DataLoader
train_json = "train.json"
image_dir = "images"
dataset = ICLEVRDataset(image_dir, train_json, object_json="objects.json")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 建立模型
model = ConditionalUNetDiffusers(num_classes=num_classes, device=device).to(device)
ddpm = DDPM(model, timesteps=1000, device=device).to(device)
evaluator = Evaluator()

# 優化器
optimizer = torch.optim.Adam(ddpm.parameters(), lr=lr)

# 檢查是否要 Resume 訓練
start_epoch = 0
best_loss = float("inf")
if os.path.exists(save_path):
    checkpoint = torch.load(save_path, map_location=device)
    ddpm.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint.get("epoch", 0)
    best_loss = checkpoint.get("loss", float("inf"))
    print(f"✅ Resumed from epoch {start_epoch}, best loss = {best_loss:.6f}")

# 訓練迴圈
for epoch in range(start_epoch, epochs):
    ddpm.train()
    running_loss = 0.0

    for img, cond in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        img = img.to(device)
        cond = cond.to(device)

        loss = ddpm(img, cond)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img.size(0)

    avg_loss = running_loss / len(dataset)
    print(f"🎯 Epoch {epoch+1}: Loss = {avg_loss:.6f}")

    # 儲存最佳模型
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            "model": ddpm.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "loss": best_loss
        }, save_path)
        print("💾 Saved best model")