import torch

# 將 [-1, 1] 區間的張量還原為 [0, 1] 區間（用於圖片顯示與儲存）
def denormalize(x):
    return (x + 1) / 2

# 將 [0, 1] 區間轉換為 [-1, 1] 區間（用於模型輸入）
def normalize(x):
    return x * 2 - 1

# 判斷是否使用 MPS（Apple Silicon）或 CUDA

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")