#%%
# Setup

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def get_random_unit_vectors(bs, n, D):
    """Generates n random unit vectors in D-dimensional space."""
    if bs is None:
        vectors = torch.randn(n, D)
    else:
        vectors = torch.randn(bs, n, D)

    vectors = vectors / vectors.norm(dim=-1, keepdim=True)
    return vectors


def get_worst_overlap(vectors):
    """Calculates the worst overlap between pairs of vectors.
    Assumes there is a batch dimension."""
    bs, n, D = vectors.shape
    overlap = torch.einsum('bik,bjk->bij', vectors, vectors)  # Inner product
    overlap = overlap.triu(diagonal=1).reshape(bs, -1)  # Keep only upper triangle
    return overlap.abs().max(dim=-1).values  # Max overlap for each vector

def get_dot_products(vectors):
    """Calculates the mean squared overlap between pairs of vectors.
    Assumes no batch dimension"""
    n, D = vectors.shape
    overlap = torch.einsum('ik,jk->ij', vectors, vectors)  # Inner product
    mask = torch.ones(n,n, dtype=torch.bool).triu(diagonal=1)  # Upper triangle mask
    overlap = overlap[mask]  # Keep only upper triangle
    return overlap  # Mean squared overlap for each vector

#%% 
#Generate data for average worst overlap plot


bs = 30
data = []
for D in [20, 40, 80, 160]:

    mean_worst_overlap = []
    ns = [2*D, 4*D, 8*D, 16*D, 32*D, 64*D, 128*D]
    for n in ns:

        vectors = get_random_unit_vectors(bs, n, D)
        worst_overlap = get_worst_overlap(vectors)
        mean_worst_overlap.append(worst_overlap.mean().item())

    sqrt_log_n_over_D = torch.sqrt(torch.log(torch.tensor(ns))/D)

    data.append({'sqrt_log_n_over_D': sqrt_log_n_over_D, 'mean_worst_overlap': mean_worst_overlap, 'D': D})


#%%
#Plot average worst overlap

for d in data:
    plt.plot(d['sqrt_log_n_over_D'], d['mean_worst_overlap'], marker='o', label=f'D = {d["D"]}')

title = f'''Max absolut dot-product of any pair of n vectors
n = [2*D, 4*D, 8*D, 16*D, 32*D, 64*D]
Averaged over Batch Size = {bs}'''


plt.title(title)
plt.xlabel('sqrt( ln(n) / D )')
plt.ylabel('Average Largest Dot-product')
plt.grid(True)
plt.xlim(left=0)
plt.ylim(bottom=0)

plt.plot([0, 1], [0, 2], 'k--', label='y = 2x')  # Reference line for y = 2x

leg = plt.legend()
for lh in leg.legend_handles:
    lh.set_alpha(1)

plt.show()


# %%

#%% 
#Generate data for worst overlap plot


bs = 30
data = []
for D in [20, 40, 80, 160]:

    mean_worst_overlap = []
    ns = [2*D, 4*D, 8*D, 16*D, 32*D, 64*D, 128*D]
    for n in ns:

        vectors = get_random_unit_vectors(bs, n, D)
        worst_overlap = get_worst_overlap(vectors)
        mean_worst_overlap.append(worst_overlap)

    sqrt_log_n_over_D = torch.sqrt(torch.log(torch.tensor(ns))/D)

    data.append({'sqrt_log_n_over_D': sqrt_log_n_over_D, 'mean_worst_overlap': mean_worst_overlap, 'D': D})


#%%
#Plot worst overlap

for d in data:
    for i, x in enumerate(d['sqrt_log_n_over_D']):
        if i == 0:
            line = plt.plot([x]*bs, d['mean_worst_overlap'][i], 'o', label=f'D = {d["D"]}', alpha=0.15)
        else:
            plt.plot([x]*bs, d['mean_worst_overlap'][i], 'o', color=line[0].get_color(), alpha=0.15)

title = f'''Max squared dot-product of any pair of T random unit vectors
T = [2*D, 4*D, 8*D, 16*D, 32*D, 64*D, 128*D]
Runs per settings = {bs}'''


plt.title(title)
plt.xlabel('sqrt( ln(T) / D )')
plt.ylabel('Max Squared Dot-product')
plt.grid(True)
plt.xlim(left=0)
plt.ylim(bottom=0)

plt.plot([0, 1], [0, 2], 'k--', label='y = 2x')  # Reference line for y = 2x

leg = plt.legend()
for lh in leg.legend_handles:
    lh.set_alpha(1)

plt.show()
# %%
# Generate data for average squared dot product 

n = 5000

mean_squared_overlaps = []
Ds = [1,2,3,4,5, 6,7, 8,9,10]

for D in Ds:
    vectors = get_random_unit_vectors(None, n, D)
    mean_squared_overlaps.append(get_dot_products(vectors).square().mean().item())


for i, D in enumerate(Ds):
    plt.plot(1/D, mean_squared_overlaps[i], 'o', label=f'D = {D}', markersize=10)

plt.xlabel('1 / D')
plt.ylabel('Mean Squared Dot-product')
plt.grid(True)
plt.xlim(left=0)
plt.ylim(bottom=0)

plt.title(f'Mean squared dot-product of all pairs of\n{n} random unit vectors in D dimensions.')

plt.plot([0, 1], [0, 1], 'k--', label='y = x')  # Reference line for y = 4x

plt.legend()


plt.show()
# %%











D = 20
n = 5000

for D in [20, 80]:
    vectors = get_random_unit_vectors(None, n, D)
    dot_products = get_dot_products(vectors)

    plt.hist(dot_products, bins=100, density=True, label=f'D = {D}', alpha=0.3)

    # Add normal distribution curve with variance 1/D
    x = np.linspace(-1, 1, 200)
    pdf = norm.pdf(x, loc=0, scale=np.sqrt(1/D))
    plt.plot(x, pdf, label=f'N(0, 1/{D})')
plt.title(f'Distribution dot-products from all pairs of \n{n} random unit vectors in D dimensions.')
plt.xlabel('Dot-product')
plt.ylabel('Probability Density')
plt.xlim(-1, 1)
plt.legend()
plt.show()


# %%
