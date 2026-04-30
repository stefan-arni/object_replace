import torch
print(torch.backends.mps.is_available())  # should be True
device = torch.device("mps")