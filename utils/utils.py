import torch

def load_model_backend(device: str):
    """
    장치가 CPU, GPU, TPU, MPS인지에 따라 모델을 로드하는 함수
    """
    if device == "cpu":
        return torch.device("cpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")