#%% Setup
#   Setup

import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm

#Make sure networks.py and assignments.py are reloaded
import importlib, networks, assignments
importlib.reload(assignments)

from assignments import (
    maxT, 
    comp_in_sup_assignment,
    test_assignments,
    slow_test_assignments,
    expected_overlap_error,
    propability_of_overlap,
    expected_squared_overlap_error
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(f'device = {device}')

#%% Customizing the __repr__ method of torch.Tensor to save images
#   Customizing the __repr__ method of torch.Tensor to save images

try:
    original_repr 
except:
    original_repr = torch.Tensor.__repr__

def custom_repr(self):
    with torch.no_grad():
        if self.dim() == 2 and self.dtype == torch.float:
            image_data = self.cpu().numpy()
            filename = f"tensor_image_{list(self.shape)}.png"
            plt.imshow(image_data, cmap='gray', aspect='equal')
            plt.axis('off')
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.close()
        try:
            mean = self.mean().item()
        except:
            mean = "N/A"
        return f"shape={list(self.shape)}, mean={mean:.5} \n{original_repr(self)}"

torch.Tensor.__repr__ = custom_repr
#%% Resetting the __repr__ method
#   Resetting the __repr__ method
try:
    torch.Tensor.__repr__ = original_repr
except:
    pass

#%% Test
#   Test
assignments, compact_assignments = comp_in_sup_assignment(T=2000, Dod=500, S=5, device=device)
test_assignments(assignments, S=5, device=device)
# %%


print(expected_overlap_error(2000, 500, 5))
print(propability_of_overlap(2000, 500, 5))
print(expected_squared_overlap_error(2000, 500, 5))
# %%
