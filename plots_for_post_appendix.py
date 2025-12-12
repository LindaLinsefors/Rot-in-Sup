
# The code here is identical to some of the conde in investigate_W.py, 
# moved here to be easier to find. 
# 
# %%
# Imports
import torch
import matplotlib.pyplot as plt

from assignments import *
from classes_and_functions import *

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




