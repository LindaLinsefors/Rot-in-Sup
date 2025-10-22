#%% Set up
#   Set up

from code import interact
import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm

device = 'cpu'
torch.set_default_device(device)

#Make sure networks.py and assignments.py are reloaded
import importlib, assignments, classes_and_functions
importlib.reload(assignments)
importlib.reload(classes_and_functions)

from assignments import (maxT, MaxT,
                         comp_in_sup_assignment,
                         expected_overlap_error, 
                         expected_squared_overlap_error, 
                         probability_of_overlap,
                         frequency_of_overlap)

from classes_and_functions import (RotSmallCircuits_3d, 
                                   RotSmallCircuits_4d,
                                   RotSmallCircuits,
                                   CompInSup, 
                                   plot_mse_rot,
                                   plot_worst_error_rot,
                                   expected_mse_rot,
                                   plot_rot)
#%%

#T, D = 500, 1000

T, D = 200, 1000

d = 4
Dod = D // d
S = 6

L = 3
z = 4

circ = RotSmallCircuits_4d(T, b=1)
net = CompInSup(D, L, S, circ)
run = net.run(L, z=z, bs=5, capped=True)
# %%
embed = net.embed
w_correction = None
rot = circ.diff_w[:,-2:,-2:]


l=1

capped_embed = torch.einsum('tn,tm->nm',embed[l],embed[l-1])
capped_embed.clamp_(max=1.0)

above_diag = torch.triu(torch.ones(T, T), diagonal=1).bool()
every_unwanted_interaction = torch.einsum('tn,nm,um->tu', embed[l], capped_embed, embed[l-1])[above_diag].sum()
every_possible_interaction = T*(T-1)/2 * S*S
capped_corr_1 = every_unwanted_interaction/(every_possible_interaction-every_unwanted_interaction) #* 10
#print('Layer', l, 'capped_corr_1:', capped_corr_1.item())

#ces = capped_embed.sum()
#capped_corr_2 = ces/(Dod**2-ces) #Alternative correction value.
#print('Layer', l, 'capped_corr_2:', capped_corr_2.item())

if w_correction is not None:
    capped_corr_1 *= w_correction

capped_embed -= (torch.ones_like(capped_embed) - capped_embed) * capped_corr_1

rot_embed = torch.einsum('tn,tij,tm->ijnm', embed[l], rot, embed[l-1])
# %%
print('Mean Error')
torch.einsum('tn,nm,um->tu', embed[l], capped_embed, embed[l-1])[above_diag].mean().item()/S
# %%
print('Capped MSE')
torch.einsum('tn,nm,um->tu', embed[l], capped_embed, embed[l-1])[above_diag].pow(2).mean().item()/S**4

# %%
print('T(d/D)^2 = ', T*(d/D)**2)

# %%
for S in [3,4,5,6]:
    net = CompInSup(D, L, S, circ)
    embed = net.embed
    print('S =', S)
    print(torch.einsum('tn,un->tu',embed[l],embed[l])[above_diag].pow(2).mean().item()/S**2)

# %%
print('Rot MSE')
torch.einsum('tn,ijnm,um,j->tui', embed[l], rot_embed, embed[l-1], torch.tensor([0.,1.]) )[above_diag].pow(2).sum(-1).mean().item()/S**4


# %%
torch.einsum('tn,ijnm,vm->tvij', embed[l], torch.einsum('un,uij,um->ijnm', embed[l], rot, embed[l-1]), embed[l-1]).pow(2).mean().item()*2/S**2

# %%
torch.einsum('tn,un,uij,um,vm->tvij', embed[l],  embed[l], rot, embed[l-1], embed[l-1]).pow(2).mean().item()*2/S**2
# %%
torch.einsum('tn,un->tu', embed[l],  embed[l]).pow(2).mean()/S**2
# %%
d/D
# %%

torch.einsum('tn,un,uij,um,vm->tvij', embed[l],  embed[l], rot, embed[l-1], embed[l-1])[above_diag].pow(2).mean()*2/S**4

# %%


capped_embed = torch.einsum('tn,tm->nm',embed[l],embed[l-1])
capped_embed.clamp_(max=1.0)
print(torch.einsum('tn,nm,um->tu', embed[l], capped_embed, embed[l-1])[:5,:5])

capped_embed -= (torch.ones_like(capped_embed) - capped_embed) * capped_corr_1
#%%
print(torch.einsum('tn,nm,um->tu', embed[l], capped_embed, embed[l-1])[:5,:5])

# %%
(torch.einsum('tn,nm,um->tu', embed[l], capped_embed, embed[l-1])[above_diag]/S**2).mean()
# %%
(torch.einsum('tn,nm,um->tu', embed[l], capped_embed, embed[l-1])[above_diag]/S**2).pow(2).mean()

# %%
torch.einsum('tn,ijnm,um,j->tui', embed[l], rot_embed, embed[l-1], torch.tensor([0.,1.])).norm(dim=-1)[:5,:5]

# %%
# %%
rot_transmission = torch.einsum('tn,ijnm,um,j->tui', 
                                embed[l], rot_embed, embed[l-1], torch.tensor([0.,1.])
                                ).norm(dim=-1)/S**2
print(rot_transmission[:5,:5])
print(rot_transmission[above_diag].pow(2).mean().item())
# %%

emb_emb = torch.einsum('tn,un->tu', embed[l],embed[l])/S
print(emb_emb[:5,:5])
print(emb_emb[above_diag].pow(2).mean().item())


# %%
emb_4 = torch.einsum('tn,un,um,vm->tv', embed[l], embed[l], embed[l-1], embed[l-1])
print(emb_4[:5,:5])
print(emb_4.mean())
print(emb_4[above_diag].mean())
print(T*(S**2*d/D)**2)
#%%
emb_4 = torch.einsum('tn,un,um,vm->tvu', embed[l], embed[l], embed[l-1], embed[l-1])
print(emb_4[:3,:3,:3])
print(emb_4.mean())
print((S**2*d/D)**2)

# %%
emb_4_rot = torch.einsum('tn,un,um,vm,uij,j->tvui', 
                         embed[l], embed[l], embed[l-1], embed[l-1], 
                         rot, torch.tensor([0.,1.])
                         ).norm(dim=-1).sum(dim=-1)
print(emb_4_rot[:5,:5])
print(emb_4_rot.mean())
print(emb_4_rot[above_diag].mean())
print(T*(S**2*d/D)**2)
#
#%%
T, D = 200, 1000
Dod = D // d
circ = RotSmallCircuits_4d(T, b=1)
net = CompInSup(D, L, S, circ)
embed = net.embed
above_diag = torch.triu(torch.ones(T, T), diagonal=1).bool()


# %%
emb_rand = torch.einsum('tn,un,um,vm,u->tv', 
                         embed[l], embed[l], embed[l-1], embed[l-1],
                         torch.randint(0,2,size=(T,)).float() * 2 - 1
                         )
print(emb_rand[:5,:5])
print(emb_rand.pow(2).mean())
print(emb_rand[above_diag].pow(2).mean())
print(T*(S**2*d/D)**2)
# %%
T=10

mask = torch.ones(T,T,T)
for i in range(T):
    for j in range(T):
        for k in range(T):
            if i==j or j==k:
                mask[i,j,k] = 0

i = torch.arange(T).view(-1, 1, 1)
j = torch.arange(T).view(1, -1, 1)
k = torch.arange(T).view(1, 1, -1)
mask_2 = ((i != j) & (j != k)).float()

(mask_2 == mask).all()

# %%
emb_rand = torch.einsum('tn,un,um,vm,u,tuv->tv', 
                         embed[l], embed[l], embed[l-1], embed[l-1],
                         torch.randint(0,2,size=(T,)).float() * 2 - 1,
                         mask
                         )
print(emb_rand[:5,:5])
print(emb_rand.pow(2).mean())
print(emb_rand[above_diag].pow(2).mean())
print(T*(S**2*d/D)**2)
# %%














#%%
# Setup D and T

DTs = [(800, 200), (800, 300),
       (1000, 200), (1000, 300), (1000, 400), (1000, 500),
       (1200, 200), (1200, 300), (1200, 400), (1200, 500), (1200, 600), (1200, 700),
       (1400, 200), (1400, 400), (1400, 600), (1400, 800), (1400, 1000),
       (1600, 300), (1600, 500), (1600, 700), (1600, 900), (1600, 1100), (1600, 1300),
       (1800, 300), (1800, 500), (1800, 700), (1800, 900), (1800, 1100), (1800, 1300), (1800, 1500), (1800, 1700),
       (2000, 200), (2000, 400), (2000, 600), (2000, 800), (2000, 1000), (2000, 1200), (2000, 1400), (2000, 1600), (2000, 1800), (2000, 2000),
       ]
D_colors = {800:'C0', 1000:'C1', 1200:'C2', 1400:'C3', 1600:'C4', 1800:'C5', 2000:'C6'}
S_colors = {1:'C0', 2:'C1', 3:'C2', 4:'C3', 5:'C4', 6:'C5', 7:'C6', 8:'C7', 9:'C8', 10:'C9'}

L = 2
d = 4

#%%
'''
mask_T = {}
for T in [200,300,400,500]:
    mask_T[T] = torch.ones(T,T,T)
    for i in range(T):
        for j in range(T):
            for k in range(T):
                if i==j or j==k:
                    mask_T[T][i,j,k] = 0
'''

#
#%%
# Generate capped data
capped_data = {}
capped_data_variant = {}
capped_data_variant_2 = {}
Ds = {}
Ts = {}

for S in [1,2,3,4,5,6,7,8,9,10]:
    capped_data[S] = {'x':[], 'y':[]}
    capped_data_variant[S] = {'x':[], 'y':[]}
    capped_data_variant_2[S] = {'x':[], 'y':[]}
    Ds[S] = []
    Ts[S] = []

    for D, T in DTs:
        circ = RotSmallCircuits_4d(T, b=1)
        try:
            net = CompInSup(D, L, S, circ)
        except:
            continue

        Ds[S].append(D)
        Ts[S].append(T)

        embed = net.embed
        w_correction = None

        l=1

        capped_embed = torch.einsum('tn,tm->nm',embed[l],embed[l-1])
        capped_embed.clamp_(max=1.0)

        above_diag = torch.triu(torch.ones(T, T), diagonal=1).bool()
        every_unwanted_interaction = torch.einsum('tn,nm,um->tu', embed[l], capped_embed, embed[l-1])[above_diag].sum()
        every_possible_interaction = T*(T-1)/2 * S*S
        capped_corr_1 = every_unwanted_interaction/(every_possible_interaction-every_unwanted_interaction) 

        capped_embed -= (torch.ones_like(capped_embed) - capped_embed) * capped_corr_1

        W = capped_embed/S

        a = torch.einsum('tn,nm,vm->tv', embed[l], W, embed[l-1])/S
        b = torch.einsum('tn,vn->tv', embed[l], embed[l])/S 
        c = torch.einsum('tm,vm->tv', embed[l-1], embed[l-1])/S

        y = a[above_diag].pow(2).mean()
        x = T*(d/D-1/(S*T))**2 + 2*(d/D - 1/(S*T))

        capped_data[S]['x'].append(x)
        capped_data[S]['y'].append(y)

        y = (a-b-c)[above_diag].pow(2).mean()
        x = T*(d/D-1/(S*T))**2
        capped_data_variant[S]['x'].append(x)
        capped_data_variant[S]['y'].append(y)

        y = (a-c)[above_diag].pow(2).mean()
        x = T*(d/D-1/(S*T))**2 + (d/D - 1/(S*T))
        capped_data_variant_2[S]['x'].append(x)
        capped_data_variant_2[S]['y'].append(y)

# %%
# plot capped data
plt.title('capped')

for S in [1,2,3,4,5,6,7,8,9,10]:
    x = [T*(d/D - 1/(S*T))**2 + 2*(d/D - 1/(S*T)) for D, T in zip(Ds[S], Ts[S])]
    plt.scatter(x, capped_data[S]['y'], label='S='+str(S), color=S_colors[S])

plt.axline((0, 0), slope=1, color='black', linestyle='--', linewidth=1)

plt.grid(True)
plt.legend()
plt.show()

for S in [1,2,3,4,5,6,7,8,9,10]:
    x = [T*(d/D - 1/(S*T))**2 + (d/D - 1/(S*T)) for D, T in zip(Ds[S], Ts[S])]
    plt.scatter(capped_data_variant[S]['x'], capped_data_variant[S]['y'], label='S='+str(S), color=S_colors[S])

plt.axline((0, 0), slope=1, color='black', linestyle='--', linewidth=1)

plt.grid(True)
plt.legend()
plt.show()

# %%

for S in [1,2,3,4,5,6,7,8,9,10]:
    x = [T*(d/D - 1/(S*T))**2 + 1*(d/D - 1/(S*T)) for D, T in zip(Ds[S], Ts[S])]
    plt.scatter(x, capped_data_variant_2[S]['y'], label='S='+str(S), color=S_colors[S])

plt.axline((0, 0), slope=1, color='black', linestyle='--', linewidth=1)

plt.grid(True)
plt.legend()
plt.show()


# %%
# Generate rand data

rand_data = {}
rand_data_variant = {}
rand_data_variant_2 = {}
Ds = {}
Ts = {}

for S in [1,2,3,4,5,6,7,8,9,10]:
    rand_data[S] = {'x':[], 'y':[]}
    rand_data_variant[S] = {'x':[], 'y':[]}
    rand_data_variant_2[S] = {'x':[], 'y':[]}
    Ds[S] = []
    Ts[S] = []

    for D, T in DTs:
        circ = RotSmallCircuits_4d(T, b=1)
        try:    
            net = CompInSup(D, L, S, circ)
        except:
            continue

        Ds[S].append(D)
        Ts[S].append(T)

        embed = net.embed
        above_diag = torch.triu(torch.ones(T, T), diagonal=1).bool()
        rand = torch.randint(0,2,size=(T,)).float() * 2 - 1

        W = torch.einsum('un,um,u->nm', embed[l], embed[l-1], rand)/S

        a = torch.einsum('tn,nm,vm->tv', embed[l], W, embed[l-1])/S
        b = torch.einsum('tn,vn,v->tv', embed[l], embed[l], rand)/S 
        c = torch.einsum('tm,vm,t->tv', embed[l-1], embed[l-1], rand)/S

        y = a[above_diag].pow(2).mean()
        x = T*(d/D-1/(S*T))**2 + 2*(d/D - 1/(S*T))

        rand_data[S]['x'].append(x)
        rand_data[S]['y'].append(y)

        y = (a-b-c)[above_diag].pow(2).mean()
        x = T*(d/D-1/(S*T))**2
        rand_data_variant[S]['x'].append(x)
        rand_data_variant[S]['y'].append(y)

        y = (a-c)[above_diag].pow(2).mean()
        x = T*(d/D-1/(S*T))**2 + (d/D - 1/(S*T))
        rand_data_variant_2[S]['x'].append(x)
        rand_data_variant_2[S]['y'].append(y)



# %%
# plot rand data
plt.title('rand')

for S in [1,2,3,4,5,6,7,8,9,10]:
    x = [T*(d/D - 1/(S*T))**2 + 2*(d/D - 1/(S*T)) for D, T in zip(Ds[S], Ts[S])]
    plt.scatter(x, rand_data[S]['y'], label='S='+str(S), color=S_colors[S])
plt.axline((0, 0), slope=1, color='black', linestyle='--', linewidth=1)
plt.grid(True)
plt.legend()
plt.show()


for S in [1,2,3,4,5,6,7,8,9,10]:
    plt.scatter(rand_data_variant[S]['x'], rand_data_variant[S]['y'], marker='o', label='S='+str(S), color=S_colors[S])
plt.axline((0, 0), slope=1, color='black', linestyle='--', linewidth=1)
plt.grid(True)
plt.legend()
plt.show()

for S in [1,2,3,4,5,6,7,8,9,10]:
    plt.scatter(rand_data_variant_2[S]['x'], rand_data_variant_2[S]['y'], marker='o', label='S='+str(S), color=S_colors[S])
plt.axline((0, 0), slope=1, color='black', linestyle='--', linewidth=1)
plt.grid(True)
plt.legend()

plt.show()


# %%
# Generate combined data

combined_data = {}
for S in [1,2,3,4,5,6,7,8,9,10]:
    combined_data[S] = {'x':[], 'y':[]}

    for D, T in DTs:
        circ = RotSmallCircuits_4d(T, b=1)
        try:
            net = CompInSup(D, L, S, circ)
        except:
            continue

        embed = net.embed
        w_correction = None
        rand = torch.randint(0,2,size=(T,)).float() * 2 - 1

        l=1

        capped_embed = torch.einsum('tn,tm->nm',embed[l],embed[l-1])
        capped_embed.clamp_(max=1.0)

        above_diag = torch.triu(torch.ones(T, T), diagonal=1).bool()
        every_unwanted_interaction = torch.einsum('tn,nm,um->tu', embed[l], capped_embed, embed[l-1])[above_diag].sum()
        every_possible_interaction = T*(T-1)/2 * S*S
        capped_corr_1 = every_unwanted_interaction/(every_possible_interaction-every_unwanted_interaction) 

        capped_embed -= (torch.ones_like(capped_embed) - capped_embed) * capped_corr_1

        W = capped_embed/S + torch.einsum('un,um,u->nm', embed[l], embed[l-1], rand)/S

        a = torch.einsum('tn,nm,vm->tv', embed[l], W, embed[l-1])/S
        c = torch.einsum('tm,vm,t->tv', embed[l-1], embed[l-1], (rand+1) )/S

        y = (a-c)[above_diag].pow(2).mean()
        x = 2*T*(d/D-1/(S*T))**2 + 2*(d/D - 1/(S*T))
        combined_data[S]['x'].append(x)
        combined_data[S]['y'].append(y)



# %% 
# Plot for post 

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
legend_handles = [mpatches.Patch(color=S_colors[S], label=f'S={S}') for S in S_colors] + \
                 [Line2D([0], [0], marker='o', linestyle='None', color='gray', label=r'$T>D/d$'),
                  Line2D([0], [0], marker='x', linestyle='None', color='gray', label=r'$T\leq D/d$'),
                  Line2D([0], [0], marker='None', linestyle='--', color='black', label='y=x')]



for j, data in enumerate([rand_data_variant_2, capped_data_variant_2, combined_data]):
    for S in [1,2,3,4,5,6,7,8,9,10]:
        for i, x in enumerate(data[S]['x']):
            y = data[S]['y'][i]
            D = Ds[S][i]
            T = Ts[S][i]

            if d*T > D:
                marker='o'
            else:
                marker='x'

            #x = T*(d/D - 1/(S*T))**2 + (d/D - 1/(S*T))
            plt.plot(x, y, marker=marker, color=S_colors[S])

    plt.axline((0, 0), slope=1, color='black', linestyle='--', linewidth=1, label='Theory')
    plt.grid(True)

    plt.xlabel(r'$T(\frac{d}{D} - \frac{1}{S*T})^2 + (\frac{d}{D} - \frac{1}{S*T})$')
    plt.ylabel(r'$\underset{v\neq t}{\mathbb{E}}\left[\left(\dfrac{1}{S}\left<{e^l_t}\right|W^l\left|{e_v^{l-1}}\right>-\dfrac{1}{S}\left<{e^{l-1}_t|e^{l-1}_t}\right>w_t\right)^2\right]$')

    if j == 0:
        plt.title(r'$\bar{w}=0$ and $\Delta w_u = \text{[1 or -1]}_u$')
    elif j == 1:
        plt.title(r'$\bar{w}=1$ and $\Delta w_u = 0$')
    elif j == 2:
        plt.title(r'$\bar{w}=1$ and $\Delta w_u = \text{[1 or -1]}_u$')
        plt.xlabel(r'$2\left(T(\frac{d}{D} - \frac{1}{S*T})^2 + (\frac{d}{D} - \frac{1}{S*T})\right)$')


    plt.legend(handles=legend_handles)
    plt.show()


















# %%
for S in [1,2,3,4,5,6,7,8,9,10]:
    x = [T*(d/D - 1/(S*T))**2 for D, T in zip(Ds[S], Ts[S])]    
    plt.scatter(x, rand_data_variant_2[S]['y'], marker='x', label='S='+str(S), color=S_colors[S])

plt.axline((0, 0), slope=1, color='black', linestyle='--', linewidth=1)
plt.grid(True)
plt.legend()
plt.xlabel('T(S^2d/D - S/T)^2')

plt.show()

# %%
# Generate rand & mask data

rand_mask_data = {}

for S in [1,2,3,4,5,6,7,8,9,10]:
    rand_mask_data[S] = {'x':[], 'y':[]}

    for D, T in DTs:
        circ = RotSmallCircuits_4d(T, b=1)
        try:    
            net = CompInSup(D, L, S, circ)
        except:
            continue

        embed = net.embed
        above_diag = torch.triu(torch.ones(T, T), diagonal=1).bool()


        y = (torch.einsum('tn,un,tuv,um,vm,u->tv', 
                          embed[l], embed[l], mask_T[T], embed[l-1], embed[l-1],
                          torch.randint(0,2,size=(T,)).float() * 2 - 1
                         )[above_diag]/S**2).pow(2).mean()
        x = T*(d/D)**2

        rand_mask_data[S]['x'].append(x)
        rand_mask_data[S]['y'].append(y)
# %%
# plot rand & mask data
plt.title('rand & mask')

for S in [1,2,3,4,5,6,7,8,9,10]:
    plt.scatter(rand_mask_data[S]['x'], rand_mask_data[S]['y'], label='S='+str(S), color=S_colors[S])

plt.axline((0, 0), slope=2, color='black', linestyle='--', linewidth=1)
plt.axline((0, 0), slope=1, color='black', linestyle='--', linewidth=1)
plt.grid(True)
plt.legend()
plt.show()

# %%

plt.title('rand & mask')

for S in [1,2,3,4,5,6,7,8,9,10]:
    plt.scatter(rand_data[S]['x'], rand_data[S]['y'],     label='S='+str(S), color=S_colors[S], alpha=0.6)
    #plt.scatter(capped_data[S]['x'], capped_data[S]['y'],  color=S_colors[S], alpha=0.6, edgecolor='black')
    for D, T in DTs:
        plt.scatter([T*(d/D)**2], [4*T*((d/D)-1/(S*T))**2], color=S_colors[S], marker='x', alpha=0.6)

plt.axline((0, 0), slope=2, color='black', linestyle='--', linewidth=1)
plt.axline((0, 0), slope=1, color='black', linestyle='--', linewidth=1)
plt.xlabel('T(d/D)^2')
plt.grid(True)
plt.legend()
plt.show()
# %%



dot_data = {}
dot_square_data = {}

for S in [1,2,3,4,5,6,7,8,9,10]:
    dot_data[S] = {'x':[], 'y':[]}
    dot_square_data[S] = {'x':[], 'y':[]}

    for D, T in DTs:
        circ = RotSmallCircuits_4d(T, b=1)
        try:    
            net = CompInSup(D, L, S, circ)
            #print(f'D={D}, T={T}, S={S}')
        except:
            continue

        embed = net.embed
        above_diag = torch.triu(torch.ones(T, T), diagonal=1).bool()


        dot = torch.einsum('tn,vn->tv', embed[l], embed[l])[above_diag]/S
        x = (d/D)**2

        dot_data[S]['x'].append(S*d/D - 1/T)
        dot_data[S]['y'].append(dot.mean())
        dot_square_data[S]['x'].append(d/D - 1/(T*S))
        dot_square_data[S]['y'].append(dot.pow(2).mean())

# %%
for S in [1,2,3,4,5,6,7,8,9,10]:
    plt.scatter(dot_data[S]['x'], dot_data[S]['y'], label='S='+str(S), color=S_colors[S])
    plt.grid(True)
    plt.axline((0, 0), slope=1, color='black', linestyle='--', linewidth=1)
plt.show()

for S in [1,2,3,4,5,6,7,8,9,10]:
    plt.scatter(dot_square_data[S]['x'], dot_square_data[S]['y'], label='S='+str(S), color=S_colors[S])
    plt.grid(True)
    plt.axline((0, 0), slope=1, color='black', linestyle='--', linewidth=1)
plt.show()
# %%

S = 6
for DT in DTs:
    D, T = DT
    print (f'D={D}, T={T}')
    print (f'D/d - 1/(T*S) = {D/d - 1/(T*S)}')
# %%

D=1000; T=500; S=6

circ = RotSmallCircuits_4d(T, b=1)

net = CompInSup(D, L, S, circ)
print(f'D={D}, T={T}, S={S}')


embed = net.embed
above_diag = torch.triu(torch.ones(T, T), diagonal=1).bool()


dot = torch.einsum('tn,vn->tv', embed[l], embed[l])[above_diag]/S
x = (d/D)**2

dot_data[S]['x'].append(S*d/D - 1/T)
dot_data[S]['y'].append(dot.mean())
dot_square_data[S]['x'].append(d/D - 1/(T*S))
dot_square_data[S]['y'].append(dot.pow(2).mean())
# %%




# %%
rand_data_same = {}

Ds = {}
Ts = {}

for S in [1,2,3,4,5,6,7,8,9,10]:
    rand_data_same[S] = {'x':[], 'y':[]}
    Ds[S] = []
    Ts[S] = []

    for D, T in DTs:
        circ = RotSmallCircuits_4d(T, b=1)
        try:    
            net = CompInSup(D, L, S, circ)
        except:
            continue

        Ds[S].append(D)
        Ts[S].append(T)

        embed = net.embed
        above_diag = torch.triu(torch.ones(T, T), diagonal=1).bool()
        rand = torch.randint(0,2,size=(T,)).float() * 2 - 1

        mask = torch.ones(T, T)
        mask.fill_diagonal_(0)

        y = (torch.einsum('tn,un,um,tm,u,tu->t', 
                          embed[l], embed[l], embed[l-1], embed[l-1], rand, mask
                         )/S**2).pow(2).mean()
        x = T*(d/D)**2

        rand_data_same[S]['x'].append(x)
        rand_data_same[S]['y'].append(y)

#%%

rand_data_diff_1 = {}
rand_data_diff_2 = {}

Ds = {}
Ts = {}

for S in [1,2,3,4,5,6,7,8,9,10]:
    rand_data_diff_1[S] = {'x':[], 'y':[]}
    rand_data_diff_2[S] = {'x':[], 'y':[]}
    Ds[S] = []
    Ts[S] = []

    for D, T in DTs:
        circ = RotSmallCircuits_4d(T, b=1)
        try:    
            net = CompInSup(D, L, S, circ)
        except:
            continue

        Ds[S].append(D)
        Ts[S].append(T)

        embed = net.embed
        above_diag = torch.triu(torch.ones(T, T), diagonal=1).bool()
        rand = torch.randint(0,2,size=(T,)).float() * 2 - 1

        mask = torch.ones(T, T)
        mask.fill_diagonal_(0)

        y = (torch.einsum('tn,un,um,um,u->tu', 
                          embed[l], embed[l], embed[l-1], embed[l-1], rand
                         )/S**2)[above_diag].pow(2).mean()

        rand_data_diff_1[S]['y'].append(y)

        y = (torch.einsum('tn,tn,tm,um,u->tu', 
                          embed[l], embed[l], embed[l-1], embed[l-1], rand
                         )/S**2)[above_diag].pow(2).mean()

        rand_data_diff_2[S]['y'].append(y)

# %%


for S in [1,2,3,4,5,6,7,8,9,10]:
    x = [T*(d/D - 1/(S*T))**2 for D, T in zip(Ds[S], Ts[S])]   
    plt.scatter(x, rand_data_same[S]['y'], label='S='+str(S), color=S_colors[S])
plt.axline((0, 0), slope=2, color='black', linestyle='--', linewidth=1)
plt.axline((0, 0), slope=1, color='black', linestyle='--', linewidth=1)
plt.grid(True)
plt.legend()
ylim = plt.ylim()
xlim = plt.xlim()
plt.show()

# %%


for S in [1,2,3,4,5,6,7,8,9,10]:
    x = [(d/D - 1/(S*T)) for D, T in zip(Ds[S], Ts[S])]   
    plt.scatter(x, rand_data_diff_1[S]['y'], label='S='+str(S), color=S_colors[S])
plt.axline((0, 0), slope=2, color='black', linestyle='--', linewidth=1)
plt.axline((0, 0), slope=1, color='black', linestyle='--', linewidth=1)
plt.grid(True)
plt.legend()
ylim = plt.ylim()
xlim = plt.xlim()
plt.show()

for S in [1,2,3,4,5,6,7,8,9,10]:
    x = [(d/D - 1/(S*T)) for D, T in zip(Ds[S], Ts[S])]   
    plt.scatter(x, rand_data_diff_2[S]['y'], label='S='+str(S), color=S_colors[S])
plt.axline((0, 0), slope=2, color='black', linestyle='--', linewidth=1)
plt.axline((0, 0), slope=1, color='black', linestyle='--', linewidth=1)
plt.grid(True)
plt.legend()
ylim = plt.ylim()
xlim = plt.xlim()
plt.show()
# %%

for S in [1,2,3,4,5,6,7,8,9,10]:
    for y1, y2 in zip(rand_data_diff_1[S]['y'], rand_data_diff_2[S]['y']):
        print(y1 - y2)

# %%
