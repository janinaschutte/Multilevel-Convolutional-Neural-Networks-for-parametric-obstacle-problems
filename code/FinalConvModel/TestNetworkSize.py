import numpy as np
import torch

from Model import BigConvModel


# target num params ~7 mil for level 7
model = BigConvModel(7, 7, None, None, None, None)

num_params = sum([x.numel() for x in model.parameters() if x.requires_grad])
print(num_params)
