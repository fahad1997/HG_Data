import torch
try:
    from torch.distributed.tensor import DTensor
    print("DTensor imported from torch.distributed.tensor")
except ImportError:
    print("DTensor not found in torch.distributed.tensor")

try:
    from torch.distributed._tensor import DTensor
    print("DTensor imported from torch.distributed._tensor")
except ImportError:
    print("DTensor not found in torch.distributed._tensor")


print(torch.__version__)

