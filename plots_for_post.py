# This notebook was used to generate the plots for our LW post.
# All plots in the post comes from here, but not all plots here
# are in the post.
# 
# If you run this code yourself, you will find that plot will not 
# look exactly the same as in the post. This is becasue there are
# random elements in the code, and I have not set any random seed.
#
# %% 
# Set up

from code import interact
import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm

# Unsing cpu becasue this code is more memory intensive 
# than compute intensive
device = 'cpu' 
torch.set_default_device(device) 

# Make sure networks.py and assignments.py are reloaded
# when this sell is run, in case I made changes to them
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
                                   plot_rot,
                                   plot_just_true_rotations,)
# %%
# Create network

T = 557 # Number of circuits
D = 1000 # Neurons per layer in the large netowrk
d = 4 # Neurons per layer in the small circuits
Dod = D // d # D/d : This number is used a lot
S = 6 # Embeding redundancy: How many netowrk neurns are used to embed each circuit neuron.

L = 5 # Number of layers
z = 1 # Number of active circuits

circ = RotSmallCircuits_4d(T, b=1) # Create T small circuits
net = CompInSup(D, L, S, circ) # Create the large network
# %% ############################################################
# Plot rotations

# This and the next few cells produces plot of the rotated vectors
# for sampled circuits, for z=1, z=2, and z=3. Comparing the true
# values to the estimated ones extracted from the large network.

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
# Figure 1 in LW post
rows=2
cols=5
bs=rows*cols
run = net.run(L, z=1, bs=bs, capped=True)
plot_rot(run, rows=rows, cols=cols, 
         colors=['green', 'purple'])

# %%
# Figure 2 in LW post
plot_just_true_rotations(run, rows=1, cols=5, colors=['green'])


# Plot step function
# Figure 3 in LW post

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
# Figure 4 in LW post
rows=1
cols=5
bs=rows*cols

run = net.run(L, z=1, bs=bs, capped=True)
plot_rot(run, rows=rows, cols=cols, 
         colors=['green', 'purple'], label='Estimated \nwith z=1')

run = net.run(L, z=2, bs=bs, capped=True)
plot_rot(run, rows=rows, cols=cols, 
         colors=['green', 'red'], label='Estimated \nwith z=2')

run = net.run(L, z=3, bs=bs, capped=True)
plot_rot(run, rows=rows, cols=cols, 
         colors=['green', 'orange'], label='Estimated \nwith z=3')


# %% 
# Generate runs

# The next few plots are based on statisitcs of may runs. 
# This cell prepares generate the data for these plots.

L = 5 # Number of layers
normal_runs = []
normal_labels = []
bs = T*100 # Batch size, i.e. number of independent runs.

# z=3, i.e. 3 simultanious active circuits.
# Runs the network for T*100 randomly sampled tripples of active circuits
run = net.run(L, z=3, bs=bs, capped=True)
normal_runs.append(run)
normal_labels.append('z=3')

# z=2, i.e. 2 simultanious active circuits.
# Runs the network for T*100 randomly sampled pairs of active circuits
run = net.run(L, z=2, bs=bs, capped=True)
normal_runs.append(run)
normal_labels.append('z=2')

# z=1, i.e. only one active ciruit per forward pass.
# Runs the network T times, each time with a different active circuit.
run = net.run(L, z=1, bs=T, capped=True, active_circuits=torch.arange(T).reshape(T,1))
normal_runs.append(run)
normal_labels.append('z=1')

#%%
# Plot Mean Squared and Worst Case Errors
importlib.reload(classes_and_functions)
from classes_and_functions import plot_mse_rot, plot_worst_error_rot

title=f'D={D}, D/d={Dod}, T={T}, S={S}'
colors = ['orange', 'red', 'purple']
plot_mse_rot(        L, normal_labels, normal_runs, title=title, colors=colors, figsize=(11,4))
plot_worst_error_rot(L, normal_labels, normal_runs, title=title, colors=colors, figsize=(11,4))


#%%
# Same plots as the ones from previous cell, but arranged differently
# Figures 5 and 6 in LW post

mse_on = []
mse_x = []
mse_on_inactive = []
mse_x_inactive = []
worst_error_on = []
worst_error_x = []
worst_error_on_inactive = []
worst_error_x_inactive = []

colors = ['orange', 'red', 'purple']

for run in normal_runs:
    mse_on.append((run.on - run.est_on).pow(2).mean(dim=(1, 2)).cpu().numpy())
    mse_x.append((run.x - run.est_x).pow(2).mean(dim=(1, 2)).sum(dim=-1).cpu().numpy())
    mse_on_inactive.append((run.est_inactive_on).pow(2).mean(dim=(1, 2)).cpu().numpy())
    mse_x_inactive.append((run.est_inactive_x).pow(2).mean(dim=(1, 2)).sum(dim=-1).cpu().numpy())
    worst_error_on.append((run.on - run.est_on).abs().amax(dim=(1, 2)).cpu().numpy())
    worst_error_x.append((run.x - run.est_x).norm(dim=-1).amax(dim=(1, 2)).cpu().numpy())
    worst_error_on_inactive.append((run.est_inactive_on).abs().amax(dim=(1, 2)).cpu().numpy())    
    worst_error_x_inactive.append((run.est_inactive_x).norm(dim=-1).amax(dim=(1, 2)).cpu().numpy())

for ptype in ['x', 'on']:
    fig = plt.figure(figsize=(8,8))
    
    if ptype == 'on':
        fig.suptitle(f'On-Indicators', fontsize=14)
    else:
        fig.suptitle(f'Rotating Vectors', fontsize=14)

    plt.subplot(2, 2, 1)
    for i, label in enumerate(normal_labels):
        if ptype == 'on':
            plt.plot(mse_on_inactive[i], marker='o', label=label, color=colors[i])
        else:
            plt.plot(mse_x_inactive[i], marker='o', label=label, color=colors[i])
    plt.title('Inactive Circuits')
    plt.ylabel('Mean Squared Error')

    plt.subplot(2, 2, 2)
    for i, label in enumerate(normal_labels):
        if ptype == 'on':
            plt.plot(mse_on[i], marker='o', label=label, color=colors[i])
        else:
            plt.plot(mse_x[i], marker='o', label=label, color=colors[i])
    plt.title('Active Circuits')
    plt.ylabel('Mean Squared Error')

    plt.subplot(2, 2, 3)
    for i, label in enumerate(normal_labels):
        if ptype == 'on':
            plt.plot(worst_error_on_inactive[i], marker='o', label=label, color=colors[i])
        else:
            plt.plot(worst_error_x_inactive[i], marker='o', label=label, color=colors[i])
    plt.title('Inactive Circuits')
    plt.ylabel('Worst Case Absolute Error')

    plt.subplot(2, 2, 4)
    for i, label in enumerate(normal_labels):
        if ptype == 'on':
            plt.plot(worst_error_on[i], marker='o', label=label, color=colors[i])
        else:
            plt.plot(worst_error_x[i], marker='o', label=label, color=colors[i])
    plt.title('Active Circuits')
    plt.ylabel('Worst Case Absolute Error')

    for j in range(4):
        plt.subplot(2, 2, j+1)
        plt.xlabel('Layer')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

# %%
# Fraction of active neurons
# Figure 7 in LW post

fig = plt.figure(figsize=(11,4))
epsilon = 1e-4

for j, run in enumerate(normal_runs):
    for i in range(4):
        A = run.A[:,:,i*Dod:(i+1)*Dod]  # (L, bs, D) -> (L, bs, Dod)
        #A = run.A[:,0,i*Dod:(i+1)*Dod]  # (L, bs, D) -> (L, bs, Dod)

        z=3-j
        plt.subplot(1, 4, i+1)
        plt.plot((A>epsilon).float().mean(dim=(1,2)), 'o-', label=normal_labels[j], color=colors[j])
        #plt.plot((A>epsilon).float().mean(dim=(1,)), 'o-', label=normal_labels[j], color=colors[j])


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

title = '' # f'D={D}, D/d={Dod}, T={T}, S={S}'
fig.text(0.5, -0.05, title, ha='center', va='bottom', fontsize=16)

plt.tight_layout()
plt.show()



# %% ####################################################################
# Here follows more experimentations, not directly shown in the post.
#########################################################################


# Generate more stable runs?

# Generate a new network where the amount of inter-circuit supression
# is increased by 15%, to see if this is better.

w_correction = 1.15
net = CompInSup(D, L, S, circ, w_correction=w_correction)

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
title=f'D={D}, D/d={Dod}, T={T}, S={S}, w_correction={w_correction}'
plot_mse_rot(        L, stable_labels, stable_runs, title=title, colors=colors, figsize=(11,4))
plot_worst_error_rot(L, stable_labels, stable_runs, title=title, colors=colors, figsize=(11,4))


# %% 
# Test different batch sizes, z=2
# Is 100*T large enough to catch the worst case errors? Seems so, yes.

runs = []
labels = []
for bs in [T, 30*T, 100*T]:
    run = net.run(L, z=2, bs=bs, capped=True)
    runs.append(run)
    labels.append(f'bs={bs}')
plot_worst_error_rot(L, labels, runs, title=f'z=2, D/d={Dod}, T={T}, S={S}')
# %%
# Test different batch sizes, z=3
# Is 100*T large enough to catch the worst case errors? Seems so, yes.

runs = []
labels = []
for bs in [T, 30*T, 100*T]:
    run = net.run(L, z=3, bs=bs, capped=True)
    runs.append(run)
    labels.append(f'bs={bs}')
plot_worst_error_rot(L, labels, runs, title=f'z=3, D/d={Dod}, T={T}, S={S}')




# %% #########################################################
# Ploting how the errors scale with D and T
##############################################################


#Checking max T for various D
#The limiting factor is the assigment algorithm

d=4
S=6

from assignments import maxT

for D in [600,  800,  1000,  1200,  1400,  1600, 1800,  2000]:
    Dod = D // d
    print(f'Maximum T for D={D}, D/d={Dod}, S={S} is {maxT(Dod, S)}')

# %%
# Alternative D and T values (not used)
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

plt.plot([T*(d/D)**2 for D, T in DTs],[T*(d/D)**2 for D, T in DTs], 'x')
plt.show()

# %% 
# Generate runs for different z, D, T

# The set of D and T values I ended up using in the post
DTs = [(800, 200), (800, 300),
       (1000, 200), (1000, 300), (1000, 400), (1000, 500),
       (1200, 200), (1200, 300), (1200, 400), (1200, 500), (1200, 600), (1200, 700),
       (1400, 200), (1400, 400), (1400, 600), (1400, 800), (1400, 1000),
       (1600, 300), (1600, 500), (1600, 700), (1600, 900), (1600, 1100), (1600, 1300),
       (1800, 300), (1800, 500), (1800, 700), (1800, 900), (1800, 1100), (1800, 1300), (1800, 1500), (1800, 1700),
       (2000, 200), (2000, 400), (2000, 600), (2000, 800), (2000, 1000), (2000, 1200), (2000, 1400), (2000, 1600), (2000, 1800), (2000, 2000),
       ]

L = 4 # Max layer
S = 6 # Embeding redundancy

w_correction = None # no modification of inter-circuit suppression
ideal = False # ideal=False is the normal run. ideal=True are not using ReLUs but instead manualy turn off Large netowrk neuorns that are not used by any active circuit.
large = False # large=True uses very large batch sizes to get more stable statistics, but takes much longer to run, and isn't nessesary.
print(f'Using w_correction={w_correction}, ideal={ideal}, large={large}')

# Variables to store information for plots
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



# %%
#Ploting MSE for various D and T, all z and l 
#Figures 8, 9, 10, and 11 in LW post


import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

#D_colors = {700:'C0', 800:'C1', 900:'C2', 1000:'C3', 1100:'C4', 1200:'C5', 1300:'C6', 1400:'C7', 1500:'C8'} 
D_colors = {800:'C0', 1000:'C1', 1200:'C2', 1400:'C3', 1600:'C4', 1800:'C5', 2000:'C6'}
Ds = sorted(list(set([D for D, T in DTs])))


d=4 # Number of neruons in each small circuit
l0 = 1 # First layer to plot (skipp layer zero)

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
                

                for j, (D, T) in enumerate(DTs):
                    if T > D/d:
                        marker = 'o'
                    else:
                        marker = '^'
                    if T*(d/D)**2 < 0.006 or True:
                        plt.plot(T*(d/D)**2, mse[j][l], marker=marker, linestyle='None', color=D_colors[D])
                        #x = T*(d/D)**2 * z * (l-1) + (d/D) * (z-1) * l + T*8*(d/D)**2 * (z-1) * (l-1) + 8*(d/D) * (z-1) * l
                        #x = T*(d/D)**2 * z * (l-1) + T*8*(d/D)**2 * (z-1) * (l-1) 

                        #x = (l-1)*(z + 16*(z-1)) * T*(d/D)**2
                        #plt.plot(x, mse[j][l], marker=marker, linestyle='None', color=D_colors[D])

                # Preserve axis limits when adding lines
                plt.plot([0], [0], linestyle='None')
                ylim = plt.ylim()
                xlim = plt.xlim()
                plt.autoscale(enable=False) 
                plt.ylim(ylim)
                plt.xlim(xlim)

                if active:
                    if a_type=='x':
                        '''
                        if l==1:
                            if z>1:
                                for D in Ds:
                                    plt.axhline(y=9*(z-1)*d/D, color=D_colors[D], linestyle=':', linewidth=1)
                        else:
                            plt.axline((0, 0), slope=(l-1)*(z + 16*(z-1)), color='black', linestyle='--', linewidth=1)
                            #plt.axline((0, 0), slope=1, color='black', linestyle='--', linewidth=1)
                        '''
                        for D in Ds:
                            T = torch.arange(100, 2000, 10)
                            x = T*(d/D)**2
                            #y = T*(d/D-1/(S*T))**2 * z * (l-1) + (d/D-1/(S*T)) * (z-1) * l + T*8*(d/D-1/(S*T))**2 * (z-1) * (l-1) + 8*(d/D-1/(S*T)) * (z-1) * l
                            y = (l-1)*T*(d/D-1/(S*T))**2*(z + (z-1)*8) + l*(d/D-1/(S*T))*(z-1)*9
                            line,  = plt.plot(x, y, color=D_colors[D], linestyle='--', linewidth=1, scalex=False, scaley=False)
                        #if l>1:
                            #plt.axline((0, 0), slope=(l-1)*(z + 16*(z-1)), color='black', linestyle='--', linewidth=1)
                    else:
                        T = torch.arange(100, 2000, 10)
                        x = T*(d/D)**2
                        y = T * 0
                        line,  = plt.plot(x, y, color='gray', linestyle='--', linewidth=1, scalex=False, scaley=False)
                else:
                    for D in Ds:
                        T = torch.arange(100, 2000, 10)
                        x = T*(d/D)**2
                        y = (d/D-1/(S*T))*z
                        line,  = plt.plot(x, y, color=D_colors[D], linestyle='--', linewidth=1, scalex=False, scaley=False)
                        


                plt.grid(True)
                plt.xlabel(r'$T\left(\dfrac{d}{D}\right)^2$')
                plt.ylabel('MSE')
                plt.plot([0], [0], linestyle='None')

                if plt.ylim()[1] < plt.xlim()[1] or ((z==1 and active and l<3) and not ideal):
                    plt.ylim(plt.xlim())
                        
        handles = [mpatches.Patch(color=D_colors[D], label=f'D={D}') for D in D_colors] + \
                    [Line2D([0], [0], marker='o', linestyle='None', color='gray', label=r'$T>D/d$'),
                    Line2D([0], [0], marker='^', linestyle='None', color='gray', label=r'$T\leq D/d$'),
                    Line2D([0], [0], color='gray', linestyle=':', label='Theory')]                        

        if active and a_type=='x':
            title = f'Active Circuits, Rotated Vector, S={S}'
        elif active and a_type=='on':
            title = f'Active Circuits, On-Indicator, S={S}'
        elif not active and a_type=='on':
            title = f'Inactive Circuits, On-Indicator, S={S}'
        else:
            title = f'Inactive Circuits, Rotated Vector, S={S}'

        if ideal:
            title += ', Ideal'
        if w_correction is not None:
            title += ', ' + r'$\chi \leftarrow$' + f'{w_correction}' + r'$\chi$'

        fig.suptitle(title, fontsize=16)



        fig.legend(handles = handles, bbox_to_anchor=(1.01, 0.95), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.show()
# %%

