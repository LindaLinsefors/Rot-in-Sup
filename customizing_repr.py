'''
Changing how tensors are displayed including shape, mean, 
and saves 2D float tensors as images.

First cells changes how tensors are displayed.
Second cell resets it back to original.

Coded up by Gurkenglas, and slighty modified by me.
'''

# %% 
# Customizing the __repr__ method of torch.Tensor to save images

import torch
import matplotlib.pyplot as plt

try:
    original_repr 
except:
    original_repr = torch.Tensor.__repr__

def custom_repr(self):
    with torch.no_grad():
        if self.dim() == 2 and self.dtype == torch.float:
            try:
                image_data = self.cpu().numpy()
                filename = f"tensor_image_{list(self.shape)}.png"
                plt.imshow(image_data, cmap='gray', aspect='equal')
                plt.axis('off')
                plt.savefig(filename, bbox_inches='tight', pad_inches=0)
                plt.close()
            except:
                pass
        try:
            mean = self.mean().item()
        except:
            mean = "N/A"
        return f"shape={list(self.shape)}, mean={mean:.5} \n{original_repr(self)}"

torch.Tensor.__repr__ = custom_repr
# %% 
# Resetting the __repr__ method
try:
    torch.Tensor.__repr__ = original_repr
except:
    pass
# %%


