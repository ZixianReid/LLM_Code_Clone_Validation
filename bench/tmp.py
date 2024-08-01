import torch
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.1])
probabilities = F.softmax(logits, dim=0)
print(probabilities)