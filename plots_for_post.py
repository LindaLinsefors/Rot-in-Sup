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
# %%
# Create network

T = 500
D = 2*T
d = 4
Dod = D // d
S = 6

L = 5
z = 1

circ = RotSmallCircuits_4d(T, b=1)
net = CompInSup(D, L, S, circ)
# %%
# Plot rotations

importlib.reload(classes_and_functions)
from classes_and_functions import plot_rot

rows=2
cols=6
bs=rows*cols

run = net.run(L, z=1, bs=bs, capped=True)
plot_rot(run, rows=rows, cols=cols, 
         title=f'z=1, D={D}, D/d={Dod}, T={T}, S={S}; Rotated Vector, Active circuits, Layers 0 to 5',
         colors=['green', 'purple'])

run = net.run(L, z=2, bs=bs, capped=True)
plot_rot(run, rows=rows, cols=cols, 
         title=f'z=2, D={D}, D/d={Dod}, T={T}, S={S}; Rotated Vector, Active circuits, Layers 0 to 5',
         colors=['green', 'red'])

run = net.run(L, z=3, bs=bs, capped=True)
plot_rot(run, rows=rows, cols=cols, 
         title=f'z=3, D={D}, D/d={Dod}, T={T}, S={S}; Rotated Vector, Active circuits, Layers 0 to 5',
         colors=['green', 'orange'])
# %%
# Generate runs
L = 5
normal_runs = []
normal_labels = []
bs = T * 100

run = net.run(L, z=3, bs=bs, capped=True)
normal_runs.append(run)
normal_labels.append('z=3')

run = net.run(L, z=2, bs=bs, capped=True)
normal_runs.append(run)
normal_labels.append('z=2')

run = net.run(L, z=1, bs=T, capped=True, active_circuits=torch.arange(T).reshape(T,1))
normal_runs.append(run)
normal_labels.append('z=1')

#%%
# Plot errors
importlib.reload(classes_and_functions)
from classes_and_functions import plot_mse_rot, plot_worst_error_rot

colors = ['orange', 'red', 'purple']
plot_mse_rot(        L, normal_labels, normal_runs, title=None, colors=colors, figsize=(11,4))
plot_worst_error_rot(L, normal_labels, normal_runs, title=f'D={D}, D/d={Dod}, T={T}, S={S}', colors=colors, figsize=(11,4))



# %%
# Generate stable runs

net = CompInSup(D, L, S, circ, w_correction=1.15)

stable_runs = []
stable_labels = []

run = net.run(L, z=3, bs=T*100, capped=True)
#run = net.run(L, z=3, bs=T, capped=True)
stable_runs.append(run)
stable_labels.append('z=3')

run = net.run(L, z=2, bs=T*100, capped=True)
#run = net.run(L, z=2, bs=T, capped=True)
stable_runs.append(run)
stable_labels.append('z=2')

run = net.run(L, z=1, bs=T, capped=True, active_circuits=torch.arange(T).reshape(T,1))
stable_runs.append(run)
stable_labels.append('z=1')

#%%
# Plot errors
importlib.reload(classes_and_functions)
from classes_and_functions import plot_mse_rot, plot_worst_error_rot

colors = ['orange', 'red', 'purple']
plot_mse_rot(        L, stable_labels, stable_runs, title=None, colors=colors, figsize=(11,4))
plot_worst_error_rot(L, stable_labels, stable_runs, title=f'D={D}, D/d={Dod}, T={T}, S={S}', colors=colors, figsize=(11,4))


# %%
# Test different batch sizes, z=2
runs = []
labels = []
for bs in [T, 30*T, 100*T]:
    run = net.run(L, z=2, bs=bs, capped=True)
    runs.append(run)
    labels.append(f'bs={bs}')
plot_worst_error_rot(L, labels, runs, title=f'z=2, D/d={Dod}, T={T}, S={S}')
# %%
# Test different batch sizes, z=3
runs = []
labels = []
for bs in [T, 30*T, 100*T]:
    run = net.run(L, z=3, bs=bs, capped=True)
    runs.append(run)
    labels.append(f'bs={bs}')
plot_worst_error_rot(L, labels, runs, title=f'z=3, D/d={Dod}, T={T}, S={S}')
# %%
# Plot step function

fig, ax = plt.subplots(1, 3, figsize=(13, 5))
ax = ax.flatten()
x = torch.tensor([-1, 0.25, 0.75, 2])

ax[0].plot(x, torch.relu(2*x-0.5), color='green')
ax[1].plot(x, torch.relu(2*x-1.5), color='green')
ax[2].plot(x, torch.relu(2*x-0.5) - torch.relu(2*x-1.5), color='green')


for a in ax:    
    a.set_ylim(0, 2)
    a.grid(True)
    a.set_yticks([0, 0.5, 1, 1.5, 2])
    a.set_xlim(-0.2, 1.2)
    a.set_xticks([0, 0.25, 0.5, 0.75, 1])


plt.show()
# %%
# %%
# Plot step function

plt.figure(figsize=(5, 3))

#plt.axes().set_aspect('equal')

x = torch.tensor([-1, 0.25, 0.75, 2])

plt.plot(x, torch.relu(2*x-0.5), label=r'$\left(a^l_t\right)_0 = ReLU(2\alpha_t^{l-1}-0.5)$', 
         color='blue', alpha=0.5, lw=6)
plt.plot(x, torch.relu(2*x-1.5), label=r'$\left(a^l_t\right)_1 = ReLU(2\alpha_t^{l-1}-1.5)$',
         color='red', alpha=0.5, lw=6)
plt.plot(x, torch.relu(2*x-0.5) - torch.relu(2*x-1.5), label=r'$\alpha_t^l = \left(a^l_t\right)_0 - \left(a^l_t\right)_1$',
         color='green', alpha=0.5, lw=6)

plt.legend()


plt.xlabel(r'$\alpha_t^{l-1} = \left(a^{l-1}_t\right)_0 - \left(a^{l-1}_t\right)_1$')

plt.ylim(-0.1, 1.6)
plt.xlim(-0.1, 1.1)

plt.grid(True)
plt.yticks([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5])
plt.xticks([0, 0.25, 0.5, 0.75, 1])

plt.show()
# %%
# Fraction of active neurons

fig = plt.figure(figsize=(11,4))
epsilon = 1e-4

for j, run in enumerate(runs):
    for i in range(4):
        A = run.A[:,:,i*Dod:(i+1)*Dod]  # (L, bs, D) -> (L, bs, Dod)
        #A = run.A[:,0,i*Dod:(i+1)*Dod]  # (L, bs, D) -> (L, bs, Dod)

        z=3-j
        plt.subplot(1, 4, i+1)
        plt.plot((A>epsilon).float().mean(dim=(1,2)), 'o-', label=labels[j], color=colors[j])
        #plt.plot((A>epsilon).float().mean(dim=(1,)), 'o-', label=labels[j], color=colors[j])


        if z==1:
            plt.plot(torch.ones(L+1)*z*S/Dod, 'x--', label=r'$\dfrac{Sd}{D}$', color=colors[j])
        else:
            plt.plot(torch.ones(L+1)*z*S/Dod, 'x--', label=f'{z}'+r'$\dfrac{Sd}{D}$', color=colors[j])

for i in range(4):
    plt.subplot(1, 4, i+1)
    if i==0:
        plt.title(r'$\left(\left|\mathbf{A}^l\right>\right)_{0:\frac{D}{d}}$')
    elif i==1:
        plt.title(r'$\left(\left|\mathbf{A}^l\right>\right)_{\frac{D}{d}:2\frac{D}{d}}$')
    elif i==2:
        plt.title(r'$\left(\left|\mathbf{A}^l\right>\right)_{\frac{D}{d}:3\frac{D}{d}}$')
    elif i==3:
        plt.title(r'$\left(\left|\mathbf{A}^l\right>\right)_{3\frac{D}{d}:D}$')
    plt.xlabel('Layer')
    plt.ylabel('Fraction Active Neurons')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 0.55)
    plt.xticks(torch.arange(L+1))

title = f'D={D}, D/d={Dod}, T={T}, S={S}'
fig.text(0.5, -0.05, title, ha='center', va='bottom', fontsize=16)

plt.tight_layout()
plt.show()
# %%
run = runs[2]  # z=1
for l in [2]:
    for i in range(2):
        plt.hist(run.pre_A[l,0,i*Dod:(i+1)*Dod].flatten(),  alpha=0.5, density=True)
    plt.axvline(x=1.5, color='black', linestyle='--', linewidth=2)
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2)
    plt.show()
# %%
# Checing chi calculations

embed = net.embed
l=3

capped_embed = torch.einsum('tn,tm->nm',embed[l],embed[l-1])
capped_embed.clamp_(max=1.0)

inverted_capped_embed = torch.ones_like(capped_embed) - capped_embed

above_diag = torch.triu(torch.ones(T, T), diagonal=1).bool()

exp = torch.einsum('tn,nm,um->tu', embed[l], capped_embed, embed[l-1])[above_diag].mean()
inv_exp = torch.einsum('tn,nm,um->tu', embed[l], inverted_capped_embed, embed[l-1])[above_diag].mean()

chi = 1/S * exp/inv_exp

W = capped_embed/S - inverted_capped_embed*chi

new_exp = torch.einsum('tn,nm,um->tu', embed[l], W, embed[l-1])[above_diag].mean()

print(exp)
print(inv_exp)
print(new_exp)

every_unwanted_interaction = torch.einsum('tn,nm,um->tu', embed[l], capped_embed, embed[l-1])[above_diag].sum()
every_possible_interaction = T*(T-1)/2 * S*S
capped_corr_1 = every_unwanted_interaction/(every_possible_interaction-every_unwanted_interaction) #* 10

print(f'chi = {chi:.10f}')
print(f'capped_corr_1/S = {capped_corr_1/S:.10f}')
# %%
d=4
S=6

from assignments import maxT

for D in [600,  800,  1000,  1200,  1400,  1600, 1800,  2000]:
    Dod = D // d
    print(f'Maximum T for D={D}, D/d={Dod}, S={S} is {maxT(Dod, S)}')

# %%

DTs = [(700, 200), 
       (800, 200), (800, 300),
       (900, 200), (900, 300), (900, 400),
       (1000, 200), (1000, 300), (1000, 400), (1000, 500),
       (1100, 200), (1100, 300), (1100, 400), (1100, 500), (1100, 600),
       (1200, 200), (1200, 300), (1200, 400), (1200, 500), (1200, 600), (1200, 700),
       (1300, 200), (1300, 300), (1300, 400), (1300, 500), (1300, 600), (1300, 700), (1300, 800),
       (1400, 200), (1400, 300), (1400, 400), (1400, 500), (1400, 600), (1400, 700), (1400, 800), (1400, 900), (1400, 1000),
       (1500, 200), (1500, 300), (1500, 400), (1500, 500), (1500, 600), (1500, 700), (1500, 800), (1500, 900), (1500, 1000), (1500, 1100),
       ]

for D, T in DTs:
    print(T*(d/D)**2)

plt.plot([T*(d/D)**2 for D, T in DTs],[T*(d/D)**2 for D, T in DTs], 'x')
plt.show()

# %% 
# Generate runs for different z, D, T

DTs = [(800, 200), (800, 300),
       (1000, 200), (1000, 300), (1000, 400), (1000, 500),
       (1200, 200), (1200, 300), (1200, 400), (1200, 500), (1200, 600), (1200, 700),
       (1400, 200), (1400, 400), (1400, 600), (1400, 800), (1400, 1000),
       (1600, 300), (1600, 500), (1600, 700), (1600, 900), (1600, 1100), (1600, 1300),
       (1800, 300), (1800, 500), (1800, 700), (1800, 900), (1800, 1100), (1800, 1300), (1800, 1500), (1800, 1700),
       (2000, 200), (2000, 400), (2000, 600), (2000, 800), (2000, 1000), (2000, 1200), (2000, 1400), (2000, 1600), (2000, 1800), (2000, 2000),
       ]

L = 4
S = 6
bs = 500

w_correction = 0 
ideal = True
large = False
print(f'Using w_correction={w_correction}, ideal={ideal}, large={large}')

DT_z_runs = {}
mse_on = {}
mse_x = {}
mse_on_inactive = {}
mse_x_inactive = {}

for z in [1, 2, 3, 4]:
    print (f'Generating runs for z={z}...')
    DT_z_runs[z] = []
    for D, T in DTs:
        circ = RotSmallCircuits_4d(T, b=1)
        net = CompInSup(D, L, S, circ, w_correction=w_correction)

        if z == 1 and large:
            net.run(L, z=z, bs=T, active_circuits = torch.arange(T).reshape(T,1))
        if z > 1 and large:
            net.run(L, z=z, bs=T*100)
        else:
            run = net.run(L, z=z, bs=500, capped=True, ideal=ideal)

        DT_z_runs[z].append(run)

    mse_on[z] = []
    mse_x[z] = []
    mse_on_inactive[z] = []
    mse_x_inactive[z] = []

    for i, run in enumerate(DT_z_runs[z]):
        mse_on[z].append((1 - run.est_on).pow(2).mean(dim=(1, 2)))
        mse_x[z].append((run.x - run.est_x).pow(2).mean(dim=(1, 2)).sum(dim=-1))

        mse_on_inactive[z].append((run.est_inactive_on).pow(2).mean(dim=(1, 2)))
        mse_x_inactive[z].append((run.est_inactive_x).pow(2).mean(dim=(1, 2)).sum(dim=-1))





# %% Plot MSE for various D and T


import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

#D_colors = {700:'C0', 800:'C1', 900:'C2', 1000:'C3', 1100:'C4', 1200:'C5', 1300:'C6', 1400:'C7', 1500:'C8'} 
D_colors = {800:'C0', 1000:'C1', 1200:'C2', 1400:'C3', 1600:'C4', 1800:'C5', 2000:'C6'}
Ds = sorted(list(set([D for D, T in DTs])))

figsize=(4, 3)
legend_handles = [mpatches.Patch(color=D_colors[D], label=f'D={D}') for D in D_colors] + \
                 [Line2D([0], [0], marker='o', linestyle='None', color='gray', label=r'$T>D/d$'),
                  Line2D([0], [0], marker='x', linestyle='None', color='gray', label=r'$T\leq D/d$')]

legend_handles_inactive = legend_handles + [Line2D([0], [0], color='gray', linestyle=':', label=r'MSE = $z\dfrac{d}{D}$')]

legend_handles_active = legend_handles + [Line2D([0], [0], color='gray', linestyle=':', label=r'MSE = $9(z-1)\dfrac{d}{D}$'),
                                          Line2D([0], [0], color='black', linestyle='--', 
                                                 label=r'MSE = $(l-1)\left( z+2^{4}(z-1)\right) T\left(\dfrac{d}{D}\right)^2$')]

legend_handles_active_z1 = legend_handles + [Line2D([0], [0], color='black', linestyle='--', label=r'MSE = $(l-1)zT\left(\dfrac{d}{D}\right)^2$')]



d=4
l0 = 1
for active in [True, False]:
    for z in [1,2,3, 4]:

        fig = plt.figure(figsize=(7, 3*(L-l0+1)))
        for l in range(l0, L+1):
            for a_type in ['on', 'x']:
                if a_type == 'on':
                    plt.subplot(L-l0+1, 2, 2*(l-l0) + 1)
                    plt.title(f'On-Indicator, Layer {l}', fontsize=11)

                    if active:
                        mse = mse_on[z]
                    else:
                        mse = mse_on_inactive[z]
                else:
                    plt.subplot(L-l0+1, 2, 2*(l-l0) + 2)
                    plt.title(f'Rotated Vector, Layer {l}', fontsize=11)

                    if active:
                        mse = mse_x[z]
                    else:
                        mse = mse_x_inactive[z]

                if active:
                    if l==1:
                        if z>1 and a_type=='x':
                            for D in Ds:
                                plt.axhline(y=9*(z-1)*d/D, color=D_colors[D], linestyle=':', linewidth=1)
                    else:
                        if z>1 and a_type=='x':
                            plt.axline((0, 0), slope=(l-1)*(z + 2**4*(z-1)), color='black', linestyle='--', linewidth=1)
                        elif z==1 and a_type=='x':
                            plt.axline((0, 0), slope=(l-1)*z, color='black', linestyle='--', linewidth=1)

                else:
                    for D in Ds:
                        plt.axhline(y=z*d/D, color=D_colors[D], linestyle=':', linewidth=1)
                

                for j, (D, T) in enumerate(DTs):
                    if T > D/d:
                        marker = 'o'
                    else:
                        marker = 'x'
                    if T*(d/D)**2 < 0.005 or True:
                        plt.plot(T*(d/D)**2, mse[j][l], marker=marker, linestyle='None', color=D_colors[D])
                    
                plt.grid(True)
                plt.xlabel(r'$T\left(\dfrac{d}{D}\right)^2$')
                plt.ylabel('MSE')
                plt.plot([0], [0], linestyle='None')

                if plt.ylim()[1] < plt.xlim()[1] or ((z==1 and active and l<3) and not ideal):
                    plt.ylim(plt.xlim())
                        
                        

        
        if active:
            if w_correction is None:
                fig.suptitle(f'Active Circuits, S={S}, z={z}', fontsize=16)
            else:
                fig.suptitle(f'Active Circuits, S={S}, z={z}, '+r'$\chi \leftarrow$' + f'{w_correction}' + r'$\chi$', fontsize=16)


            if z==1:
                handles = legend_handles_active_z1
            else:
                handles = legend_handles_active
        else:
            if w_correction is None:
                fig.suptitle(f'Inactive Circuits, S={S}, z={z}', fontsize=16)
            else:
                fig.suptitle(f'Inactive Circuits, S={S}, z={z}, '+r'$\chi \leftarrow$' + f'{w_correction}' + r'$\chi$', fontsize=16)
            handles = legend_handles_inactive

        fig.legend(handles = handles, bbox_to_anchor=(1.05, 0.95), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.show()
# %%



import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

#D_colors = {700:'C0', 800:'C1', 900:'C2', 1000:'C3', 1100:'C4', 1200:'C5', 1300:'C6', 1400:'C7', 1500:'C8'} 
D_colors = {800:'C0', 1000:'C1', 1200:'C2', 1400:'C3', 1600:'C4', 1800:'C5', 2000:'C6'}
Ds = sorted(list(set([D for D, T in DTs])))



d=4
active = True

a_type = 'on'
l0 = 1

L = 4
z_max = 4
for active in [True, False]:
    for a_type in ['on', 'x']:

        fig = plt.figure(figsize=(12,12))

        for l in range(l0, L+1):
            for z in range(1, z_max+1):

                if a_type == 'on':
                    if active:
                        mse = mse_on[z]
                    else:
                        mse = mse_on_inactive[z]
                else:
                    if active:
                        mse = mse_x[z]
                    else:
                        mse = mse_x_inactive[z]

                plt.subplot(L-l0+1, z_max, 1 + (l-l0)*z_max + (z-1))
                plt.title(f'Layer {l}, z={z}', fontsize=11)



                if active:
                    if a_type=='x':
                        if l==1:
                            if z>1:
                                for D in Ds:
                                    plt.axhline(y=9*(z-1)*d/D, color=D_colors[D], linestyle=':', linewidth=1)
                        else:
                            plt.axline((0, 0), slope=(l-1)*(z + 2**4*(z-1)), color='black', linestyle='--', linewidth=1)

                else:
                    for D in Ds:
                        plt.axhline(y=z*d/D, color=D_colors[D], linestyle=':', linewidth=1)
                

                for j, (D, T) in enumerate(DTs):
                    if T > D/d:
                        marker = 'o'
                    else:
                        marker = 'x'
                    if T*(d/D)**2 < 0.006 or True:
                        plt.plot(T*(d/D)**2, mse[j][l], marker=marker, linestyle='None', color=D_colors[D])
                    
                plt.grid(True)
                plt.xlabel(r'$T\left(\dfrac{d}{D}\right)^2$')
                plt.ylabel('MSE')
                plt.plot([0], [0], linestyle='None')

                if plt.ylim()[1] < plt.xlim()[1] or ((z==1 and active and l<3) and not ideal):
                    plt.ylim(plt.xlim())
                        
        handles = [mpatches.Patch(color=D_colors[D], label=f'D={D}') for D in D_colors] + \
                    [Line2D([0], [0], marker='o', linestyle='None', color='gray', label=r'$T>D/d$'),
                    Line2D([0], [0], marker='x', linestyle='None', color='gray', label=r'$T\leq D/d$')]                        

        if active and a_type=='x':
            title = f'Active Circuits, Rotated Vector, S={S}'
            handles += [Line2D([0], [0], color='gray', linestyle=':', label=r'MSE = $9(z-1)\dfrac{d}{D}$'),
                        Line2D([0], [0], color='black', linestyle='--', 
                            label=r'MSE = $(l-1)\left( z+2^{4}(z-1)\right) T\left(\dfrac{d}{D}\right)^2$')]
        elif active and a_type=='on':
            title = f'Active Circuits, On-Indicator, S={S}'
            
        elif not active and a_type=='on':
            title = f'Inactive Circuits, On-Indicator, S={S}'
            handles += [Line2D([0], [0], color='gray', linestyle=':', label=r'MSE = $z\dfrac{d}{D}$')]

        else:
            title = f'Inactive Circuits, Rotated Vector, S={S}'
            handles += [Line2D([0], [0], color='gray', linestyle=':', label=r'MSE = $z\dfrac{d}{D}$')]
                

        if ideal:
            title += ', Ideal'
        if w_correction is not None:
            title += ', ' + r'$\chi \leftarrow$' + f'{w_correction}' + r'$\chi$'

        fig.suptitle(title, fontsize=16)



        fig.legend(handles = handles, bbox_to_anchor=(1.01, 0.95), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.show()
# %%
