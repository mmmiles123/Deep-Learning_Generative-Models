import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from dataset import TestConditionDataset, load_obj2idx, labels_to_multihot
from model import ConditionalUNetDiffusers
from ddpm import DDPM
from evaluator import Evaluator
from utils import denormalize

def generate_images(ddpm, cond_loader, save_dir, grid_path, show_process=False, evaluator=None):
    os.makedirs(save_dir, exist_ok=True)
    all_images = []

    if show_process:
        process_labels = ["red sphere", "cyan cylinder", "cyan cube"]
        onehot = labels_to_multihot(process_labels, obj2idx)  # shape: (num_classes,)
        cond_tensor = torch.tensor(onehot, dtype=torch.float32).unsqueeze(0).repeat(8, 1).to(device)

        img, intermediates = ddpm.sample(cond_tensor, save_intermediates=True)
        process_img = ddpm.make_denoise_grid(intermediates)
        save_image(process_img, os.path.join(save_dir, "denoising_process.png"))


        img, intermediates = ddpm.sample(cond_tensor, save_intermediates=True)
        process_img = ddpm.make_denoise_grid(intermediates)
        save_image(process_img, os.path.join(save_dir, "denoising_process.png"))

    for idx, cond in enumerate(tqdm(cond_loader, desc="🔄 Generating images")):
        cond = cond.to(device)
        batch_size = cond.size(0)

        # ✅ 處理回傳為 tuple or not
        samples = ddpm.sample(cond, batch_size=batch_size)
        if isinstance(samples, tuple):
            samples = samples[0]

        samples = denormalize(samples)

        for i in range(batch_size):
            save_image(samples[i], os.path.join(save_dir, f"{idx * batch_size + i}.png"))
            all_images.append(samples[i])

    grid = make_grid(torch.stack(all_images), nrow=8)
    save_image(grid, grid_path)
    return all_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="test.json")
    parser.add_argument("--save_dir", type=str, default="images_gen/test")
    parser.add_argument("--grid_path", type=str, default="images_gen/test_grid.png")
    parser.add_argument("--show_process", action="store_true")
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_ddpm.pt")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用裝置：{device}")

    obj2idx, idx2obj = load_obj2idx("objects.json")
    num_classes = len(obj2idx)

    model = ConditionalUNetDiffusers(num_classes=num_classes, device=device).to(device)
    ddpm = DDPM(model, device=device).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ddpm.load_state_dict(checkpoint["model"])
    print(f"✅ 載入模型 checkpoint：epoch {checkpoint.get('epoch', '?')}，loss={checkpoint.get('loss', '?'):.6f}")

    evaluator = Evaluator().to(device)
    cond_dataset = TestConditionDataset(args.json, obj2idx)
    cond_loader = DataLoader(cond_dataset, batch_size=32, shuffle=False)

    all_imgs = generate_images(ddpm, cond_loader, args.save_dir, args.grid_path, args.show_process, evaluator)

    labels = torch.stack([cond for cond in cond_dataset]).to(device)
    acc = evaluator.eval(torch.stack(all_imgs).to(device), labels)
    print(f"🎯 準確率 on {args.json}：{acc:.4f}")
