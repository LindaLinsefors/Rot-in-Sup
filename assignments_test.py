#%% Setup
#   Setup

import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm

#Make sure networks.py and assignments.py are reloaded
import importlib, assignments
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

Ss = (2,3,4,5)
Dods = (100, 150, 200, 300, 400, 500)
Ts = (400,600,800,1000,1200,1400,1600,1800,2000)

overlap_frequencies = []
expected_overlap_frequencies = []
naive_expected_overlap_frequencies = []
std = []

for S in Ss:
    for Dod in Dods:
        for T in Ts:
            try:
                assignments, compact_assignments = comp_in_sup_assignment(T=T, Dod=Dod, S=S, device=device)
            except:
                #print(f"Failed for T={T}, Dod={Dod}, S={S}")
                continue
            overlap = torch.triu(torch.einsum('ti,ui->tu', assignments, assignments), diagonal=1)

            n = T*(T-1)//2

            p = overlap.sum().item() / n
            ep = propability_of_overlap(T, Dod, S)
            naive_ep = S**2 / Dod

            # print(f"T= {T}, Dod={Dod}, S={S}")
            # print(f"Overlap frequency:          {p:.5f}")
            # print(f"Expected overlap frequency: {ep:.5f}")
            # print(f"S^2/Dod:                    {naive_ep:.5f}\n")


            overlap_frequencies.append(p)
            expected_overlap_frequencies.append(ep)
            naive_expected_overlap_frequencies.append(naive_ep)
            std.append({S:S, Dod:Dod, T:T})

#%%
#Plotting vs expected overlap frequency
plt.scatter(expected_overlap_frequencies, overlap_frequencies, alpha=0.2, color='green')
plt.xlabel('Expected overlap frequency')
plt.ylabel('Overlap frequency')

#vertical line at y=x
max_value = max(max(expected_overlap_frequencies), max(overlap_frequencies))
plt.plot([0, max_value], [0, max_value], '--', color='gray')

plt.show()

#Plotting vs naive expected overlap frequency
plt.scatter(naive_expected_overlap_frequencies, overlap_frequencies, alpha=0.2, color='blue')
plt.xlabel('Naive expected overlap frequency')
plt.ylabel('Overlap frequency')

#vertical line at y=x
max_value = max(max(expected_overlap_frequencies), max(overlap_frequencies))
plt.plot([0, max_value], [0, max_value], '--', color='gray')

plt.show()
# %%
plt.scatter(overlap_frequencies, naive_expected_overlap_frequencies, 
            label='EOF = S^2/Dod', alpha=0.3, color='blue', marker='x')
plt.scatter(overlap_frequencies, expected_overlap_frequencies, 
            label='EOF = S(TS/Dod - 1)/(T - 1)', 
            alpha=0.3, color='red', marker='x')

plt.xlabel('Messured Overlap Frequency')
plt.ylabel('Expected Overlap Frequency')

#vertical line at y=x
max_value = max(max(expected_overlap_frequencies), max(naive_expected_overlap_frequencies), max(overlap_frequencies))
plt.plot([0, max_value], [0, max_value], 'k--')

plt.legend()
plt.title('Expected Overlap Frequency vs Measured Overlap Frequency')
plt.show()

# %%
