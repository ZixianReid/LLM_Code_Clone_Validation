import torch

if torch.cuda.is_available():
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(1))