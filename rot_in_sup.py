'''
Running experiments on the networks found in rot_in_sup_networks.py.

These implentations uses indicator neurons with a fixed embedding so they 
don't have to be re-calculated from layer to layer. This means less noise,
but also a less general implementation.

RotInSupNetwork_4d uses two 'on' indicators for each small network 
bringing the neurons per circuit up to d=4.

RotInSupNetwork_3d uses one 'on' indicator for each small network
bringing the neurons per circuit up to d=3, at the cost of higer error.
'''




# %% Setup  
#   Setup

######################################################################################
#   Setup
#   Immporting stuff, and setting up the device
######################################################################################

import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm
import pandas as pd

#Make sure networks.py and assignments.py are reloaded
import importlib, rot_in_sup_networks, assignments
importlib.reload(rot_in_sup_networks)
importlib.reload(assignments)

from rot_in_sup_networks import RotInSupNetwork_4d, RotInSupNetwork_3d, expected_mse_4d, expected_mse_3d
from assignments import (maxT, MaxT,
                         expected_overlap_error, 
                         expected_squared_overlap_error, 
                         propability_of_overlap)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(f'device = {device}')

#%%

######################################################################################
#   Changing what print does for torch.Tensor
#   First cell set up custom print, second cell reset it
######################################################################################

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


#%% Small test
#   Small test

######################################################################################
#   Test that the code runs OK
######################################################################################

Dod=5
S=2
T=2
L=4
z=1
bs=2

smal_test_net_4d = RotInSupNetwork_4d(Dod,T,S,device=device)
test_run_4d = smal_test_net_4d.run(L,z,bs)

if (test_run_4d.x - test_run_4d.est_x).sum().abs() > 1e-6:
    print("RotInSupNetwork_4d test failed: The output does not match the expected result.")
else:
    print("RotInSupNetwork_4d test passed: The output matches the expected result.")

smal_test_net_3d = RotInSupNetwork_3d(Dod,T,S,device=device)
test_run_3d = smal_test_net_3d.run(L,z,bs)

if (test_run_3d.x - test_run_3d.est_x).sum().abs() > 1e-6:
    print("RotInSupNetwork_3d test failed: The output does not match the expected result.")
else:
    print("RotInSupNetwork_3d test passed: The output matches the expected result.")

#%% 
######################################################################################
#   Compare Mean Squared Error (MSE) and Standard Error (STE) 
#   for various network configurations
#
#   Last cell plot the output of previously run cell
######################################################################################

#%% Compare 3d vs 4d
#   Compare 3d vs 4d
D=1200
S=5
T=1000
L=7
z=1
bs=1000

# Create lists to store results for plotting
mse_results = []
ste_results = []
labels = []

test_net = RotInSupNetwork_3d(D/3,T,S,L,balance=True, device=device)
test_run = test_net.run(L,z,bs)
e = test_run.x - test_run.est_x
mse = (e ** 2).mean((1,2)).sum((-1,))
ste = mse**0.5
mse_results.append(mse)
ste_results.append(ste)
labels.append('d=3 Network, balance=True')

test_net = RotInSupNetwork_3d(D/3,T,S,L,balance=False, device=device)
test_run = test_net.run(L,z,bs)
e = test_run.x - test_run.est_x
mse = (e ** 2).mean((1,2)).sum((-1,))
ste = mse**0.5
mse_results.append(mse)
ste_results.append(ste)
labels.append('d=3 Network, balance=False')

test_net = RotInSupNetwork_4d(D/4,T,S,device=device)
test_run = test_net.run(L,z,bs)
e = test_run.x - test_run.est_x
mse = (e ** 2).mean((1,2)).sum((-1,))
ste = mse**0.5
mse_results.append(mse)
ste_results.append(ste)
labels.append('d=4 Network')

title = f'D={D}, T={T}, S={S}, z={z}, batch size={bs}'



#%% Compare L_W=2 vs L_W=L
#   Compare L_W=2 vs L_W=L
D=1200
S=5
T=2000
L=7
z=1
bs=1000

# Create lists to store results for plotting
mse_results = []
ste_results = []
labels = []

test_net = RotInSupNetwork_3d(D/3,T,S,2,balance=True)
test_run = test_net.run(L,z,bs)
e = test_run.x - test_run.est_x
mse = (e ** 2).mean((1,2)).sum((-1,))
ste = mse**0.5
mse_results.append(mse)
ste_results.append(ste)
labels.append('d=3 Network, balance=True, L_W = 2')

test_net = RotInSupNetwork_3d(D/3,T,S,2,balance=False)
test_run = test_net.run(L,z,bs)
e = test_run.x - test_run.est_x
mse = (e ** 2).mean((1,2)).sum((-1,))
ste = mse**0.5
mse_results.append(mse)
ste_results.append(ste)
labels.append('d=3 Network, balance=False, L_W = 2')

test_net = RotInSupNetwork_3d(D/3,T,S,L,balance=True)
test_run = test_net.run(L,z,bs)
e = test_run.x - test_run.est_x
mse = (e ** 2).mean((1,2)).sum((-1,))
ste = mse**0.5
mse_results.append(mse)
ste_results.append(ste)
labels.append('d=3 Network, balance=True, L_W = L')

test_net = RotInSupNetwork_3d(D/3,T,S,L,balance=False)
test_run = test_net.run(L,z,bs)
e = test_run.x - test_run.est_x
mse = (e ** 2).mean((1,2)).sum((-1,))
ste = mse**0.5
mse_results.append(mse)
ste_results.append(ste)
labels.append('d=3 Network, balance=False, L_W = L')

title = f'D={D}, T={T}, S={S}, z={z}, batch size={bs}'



#%% Compare S for d=4
#   Compare S for d=4
D=1200
T=1000
L=7
z=1
bs=1000

# Create lists to store results for plotting
mse_results = []
ste_results = []
labels = []

for S in tqdm.tqdm(range(2,5+1)):
    test_net = RotInSupNetwork_4d(D/4,T,S)
    test_run = test_net.run(L,z,bs)
    e = test_run.x - test_run.est_x
    mse = (e ** 2).mean((1,2)).sum((-1,))
    ste = mse**0.5
    mse_results.append(mse)
    ste_results.append(ste)
    labels.append('S = ' + str(S))

title = f'D={D}, D/d={int(D/4)}, T={T}, L={L}, z={z}, batch size={bs}'


#%% Compare balance
#   Compare balance
D=1200
S=5
T=1000
L=7
z=3
bs=10000

# Create lists to store results for plotting
mse_results = []
ste_results = []
labels = []


test_net = RotInSupNetwork_3d(D/3,T,S,L,balance=True, improved_balance=True)
test_run = test_net.run(L,z,bs)
e = test_run.x - test_run.est_x
mse = (e ** 2).mean((1,2)).sum((-1,))
ste = mse**0.5
mse_results.append(mse)
ste_results.append(ste)
labels.append('balance=True, improved=True')

test_net = RotInSupNetwork_3d(D/3,T,S,L,balance=True, improved_balance=False)
test_run = test_net.run(L,z,bs)
e = test_run.x - test_run.est_x
mse = (e ** 2).mean((1,2)).sum((-1,))
ste = mse**0.5
mse_results.append(mse)
ste_results.append(ste)
labels.append('balance=True, improved=False')

test_net = RotInSupNetwork_3d(D/3,T,S,L,balance=False)
test_run = test_net.run(L,z,bs)
e = test_run.x - test_run.est_x
mse = (e ** 2).mean((1,2)).sum((-1,))
ste = mse**0.5
mse_results.append(mse)
ste_results.append(ste)
labels.append('balance=False')

title = f'D={D}, T={T}, S={S}, z={z}, batch size={bs}, \nd=3, L_W=L'




#%% Compare z values for d=4
#   Compare z values for d=4

D=1200
S=3
T=5000
L=5
bs=1000

zs = [1, 2, 3, 4, 5]

# Create lists to store results for plotting
mse_results = []
ste_results = []
labels = []
expected_mses = []

for z in tqdm.tqdm(zs):
    test_net = RotInSupNetwork_4d(D/4,T,S,device=device)
    test_run = test_net.run(L,z,bs)
    e = test_run.x - test_run.est_x
    mse = (e ** 2).mean((1,2)).sum((-1,))
    #mse = ((e ** 2).sum((-1,))**0.5).mean((1,2))
    ste = mse**0.5
    mse_results.append(mse)
    ste_results.append(ste)
    labels.append(f'z={z}')
    expected_mses.append([expected_mse_4d(T, D/4, L, z, naive=False) for L in range(L)])

title = f'D={D}, d=4, T={T}, S={S}, batch size={bs}'

plot_results_with_expected_mse(mse_results, ste_results, expected_mses, labels, title)

#%% Compare T values for d=4
#   Compare T values for d=4

D=1200
S=3
z=2
L=5
bs=1000

Ts = [1000, 2000, 3000, 4000, 5000]

# Create lists to store results for plotting
mse_results = []
ste_results = []
labels = []
expected_mses = []

for T in tqdm.tqdm(Ts):
    test_net = RotInSupNetwork_4d(D/4,T,S,device=device)
    test_run = test_net.run(L,z,bs,ideal=True)
    e = test_run.x - test_run.est_x
    mse = (e ** 2).mean((1,2)).sum((-1,))
    #mse = ((e ** 2).sum((-1,))**0.5).mean((1,2))
    ste = mse**0.5
    mse_results.append(mse)
    ste_results.append(ste)
    labels.append(f'T={T}')
    expected_mses.append([expected_mse_4d(T, D/4, L, z, naive=False) for L in range(L)])

title = f'D={D}, d=4, z={z}, S={S}, batch size={bs}'

plot_results_with_expected_mse(mse_results, ste_results, expected_mses, labels, title)


#%% Compare z values for d=3
#   Compare z values for d=3

D=1200
S=3
T=2000
L=4
bs=1000

zs = [1, 2, 3, 4, 5]

# Create lists to store results for plotting
mse_results = []
ste_results = []
labels = []
expected_mses = []

for z in tqdm.tqdm(zs):
    test_net = RotInSupNetwork_3d(D/3,T,S,L,device=device)
    test_run = test_net.run(L,z,bs)
    e = test_run.x - test_run.est_x
    mse = (e ** 2).mean((1,2)).sum((-1,))
    ste = mse**0.5
    mse_results.append(mse)
    ste_results.append(ste)
    labels.append(f'z={z}')
    expected_mses.append([expected_mse_3d(T, D/3, L, z, naive=False) for L in range(L)])


title = f'D={D}, d=3, T={T}, S={S}, batch size={bs}'

plot_results_with_expected_mse(mse_results, ste_results, expected_mses, labels, title)

#%% Plotting the results
#   Plotting the results

# Plot MSE
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
for i, mse in enumerate(mse_results):
    plt.plot(mse.cpu(), label=labels[i], marker='o')
plt.xlabel('Layer')
plt.ylabel('Mean Squared Error')
plt.title(title)
plt.grid(True)
plt.legend()
#plt.ylim(0,3)


# Plot STE
plt.subplot(1, 2, 2)
for i, ste in enumerate(ste_results):
    plt.plot(ste.cpu(), label=labels[i], marker='o')
plt.xlabel('Layer')
plt.ylabel('Standard Error')
plt.title(title)
plt.grid(True)
plt.legend()
#plt.ylim(0,3)

plt.tight_layout()
plt.show()


#%% Plotting the results with expected MSE
#   Plotting the results with expected MSE

def plot_results_with_expected_mse(mse_results, ste_results, expected_mses, labels, title):
    # Plot MSE
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    for i, mse in enumerate(mse_results):
        line, = plt.plot(mse.cpu(), label=labels[i], marker='o')
        plt.plot(expected_mses[i], linestyle='--', color=line.get_color(), marker='x')
    plt.xlabel('Layer')
    plt.ylabel('Mean Squared Error')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.ylim(0,min(3,plt.ylim()[1]))


    # Plot STE
    plt.subplot(1, 2, 2)
    for i, ste in enumerate(ste_results):
        line, = plt.plot(ste.cpu(), label=labels[i], marker='o')
        plt.plot([foo**0.5 for foo in expected_mses[i]], linestyle='--', color=line.get_color(), marker='x')
    plt.xlabel('Layer')
    plt.ylabel('Standard Error')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.ylim(0,min(3,plt.ylim()[1]))

    plt.tight_layout()
    plt.show()

#%%
######################################################################################
#   Looking for active neurons that shouldn't be active
######################################################################################

'''
Counts active neurons in the large network
Valid for layers 2 and onwards
'''


D=1200
S=5
T=1000
L=8
z=1
bs=10
d=3

if d == 4:
    net = RotInSupNetwork_4d(D/d,T,S,device=device)
if d == 3:
    net = RotInSupNetwork_3d(D/d,T,S,L,device=device)

run = net.run(L,z,bs)
other_run = net.run(L,z,bs)

A = run.A
d = run.net.d
z = run.z
L = run.L
active_circuits = run.active_circuits
Dod = run.net.Dod


if d == 4:
    L_W = 2
    assignments = torch.zeros(L_W, T, Dod, device=device)
    assignments[1] = run.net.assignments_1
    assignments[0] = run.net.assignments_2
elif d == 3:
    L_W = run.net.L_W
    assignments = run.net.assignments
else:
    Exception("Unexpected value for d")

inf_assignments = torch.zeros_like(assignments)
inf_assignments[assignments > 0] = float('inf')

A_subt = A[:,:,(d-2)*Dod:].clone()

for l in range(1,L):
    subtract = torch.einsum('btn->bn',(inf_assignments[l%L_W, active_circuits]))
    A_subt[l, :, :Dod] -= subtract
    A_subt[l, :, Dod:] -= subtract
A_subt[0] = A_subt[1]

active_neurons_that_should_not_be_active = (A_subt > 0).sum(-1)
active_neurons = (A[:,:,(d-2)*Dod:] > 0).sum(-1)
active_neurons[0] = active_neurons[1]

print(f"D={D}, S={S}, T={T}, L={L}, z={z}, d={d}")
print(f"Active neurons that should not be active: \n{active_neurons_that_should_not_be_active}")  
print(f"Active neurons: \n{active_neurons}")

# %%
active_circuits = run.active_circuits[0]
compact_assignments = run.net.compact_assignments


######################################################################################
#   Somme older plots
######################################################################################

# %%
Dod=600
D=int(4*Dod)
S=5
T=4000
L=8
N = 3600

net = RotInSupNetwork_4d(Dod,T,S)
for z in range(1,5):
   bs = N/z
   run = net.run(L,z,bs, run_name=z)

# %%
for z in range(1,5):
   error = net.run_by_name[z].est_x - net.run_by_name[z].x
   mse = (error ** 2).mean((1,2,3))
   plt.plot(mse, label=f'z={z}')
plt.xlabel('Layer')
plt.ylabel('Mean Squared Error')
plt.title(f'D={D}, D/d={Dod}, T={T}, S={S}, N={N}')
plt.legend()
plt.show()
# %%
for z in range(1,5):
   error = net.run_by_name[z].est_x - net.run_by_name[z].x
   mean_error = error.mean((1,2,3))
   plt.plot(mean_error, label=f'z={z}')
plt.xlabel('Layer')
plt.ylabel('Mean Error')
plt.title(f'D={D}, D/d={Dod}, T={T}, S={S}, N={N}')
plt.legend()
plt.show()

# %%
l=2
for z in range(1,5):
   error = net.run_by_name[z].est_x - net.run_by_name[z].x
   plt.hist(error[l].flatten().cpu().numpy(), bins=50, alpha=0.5, label=f'z={z}', density=True)
plt.title(f'D={D}, D/d={Dod}, T={T}, S={S}, N={N} : Error distribution for layer {l}')
plt.legend()
plt.show()
# %%

Dod=500
D=int(4*Dod)
T=1000
bs = 36000

netS = {}
for S in tqdm.tqdm(range(2,8)):
    netS[S] = RotInSupNetwork_4d(Dod,T,S)
    for z in range(1,5):
        run = netS[S].run(L,z,bs, run_name=z)

# %%
for z in range(1,5):

    for S in range(2,8):
        if S == 2:
           color = 'blue'
        elif S == 3:
           color = 'green'
        elif S == 4:
           color = 'orange'
        elif S == 5:
           color = 'red'
        elif S == 6:
           color = 'purple'
        elif S == 7:
           color = 'black'

        error = netS[S].run_by_name[z].est_x - netS[S].run_by_name[z].x
        mse = (error ** 2).mean((1,2,3))
        plt.plot(mse, label=f'S={S}', color=color)
    plt.xlabel('Layer')
    plt.ylabel('Mean Squared Error')
    plt.title(f'D={D}, D/d={Dod}, T={T}, z={z}, N={N}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

for z in range(1,5):
    if z == 1:
        linestyle = 'solid'
    elif z == 2:
        linestyle = 'dashed'
    elif z == 3:
        linestyle = 'dotted'

    for S in range(2,8):
        if S == 2:
           color = 'blue'
        elif S == 3:
           color = 'green'
        elif S == 4:
           color = 'orange'
        elif S == 5:
           color = 'red'
        elif S == 6:
           color = 'purple'
        elif S == 7:
           color = 'black'

        error = netS[S].run_by_name[z].est_x - netS[S].run_by_name[z].x
        mse = (error ** 2).mean((1,2,3))
        plt.plot(mse, label=f'z={z}, S={S}', color=color, linestyle=linestyle)
plt.xlabel('Layer')
plt.ylabel('Mean Squared Error')
plt.title(f'D={D}, D/d={Dod}, T={T}, N={N}')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
S=5
for z in range(1,5):
    error = netS[S].run_by_name[z].est_x - netS[S].run_by_name[z].x
    mse = (error ** 2).sum(-1).mean((1,2))
    #mse = (error ** 2).mean((1,2,3))
    print('z=',z)
    print(f'{mse[1]}')
    print((z-1)*(1+2)/Dod)
# %%
z=2
A0 = netS[S].run_by_name[z].A[0]
active_circuits = netS[S].run_by_name[z].active_circuits
assignments_1 = netS[S].assignments_1
assignments_2 = netS[S].assignments_2
Dod = netS[S].Dod
est_active = torch.einsum('bti,bi->bt', (assignments_1[active_circuits], A0[:,:Dod]))/S
mse=((est_active-1)**2).mean()
print(mse)
print((z-1)/Dod)

# %%#######################################################
#   Worst case behaviour
###########################################################

D=1200
S=5
T=1000
L=8
N=5

for n in range(N):
    active_circuits = torch.tensor(range(T), device=device).reshape(T,1)
    net = RotInSupNetwork_4d(D/4,T,S,device=device)
    run = net.run(L, active_circuits=active_circuits)

    error = (((run.x - run.est_x)**2).sum(-1)**0.5)[:,:,0]
    max_error, _ = error.max(1)
    if n == 0:
        plt.plot(max_error.cpu().numpy(), color='blue', marker='o', markersize=2, alpha=0.5,
                 label = 'd=4')
    else:
        plt.plot(max_error.cpu().numpy(), color='blue', marker='o', markersize=2, alpha=0.5)

    active_circuits = torch.tensor(range(T), device=device).reshape(T,1)
    net = RotInSupNetwork_3d(D/3,T,S,device=device)
    run = net.run(L, active_circuits=active_circuits)

    error = (((run.x - run.est_x)**2).sum(-1)**0.5)[:,:,0]
    max_error, _ = error.max(1)
    if n == 0:
        plt.plot(max_error.cpu().numpy(), color='red', marker='o', markersize=2, alpha=0.5,
                 label = 'd=3')
    else:
        plt.plot(max_error.cpu().numpy(), color='red', marker='o', markersize=2, alpha=0.5)
plt.grid()
plt.xlabel('Layer')
plt.ylabel('Max Error')
plt.title(f'D={D}, D/d={Dod}, T={T}, z=1, S={S}')
plt.legend()
plt.show()
# %%
D=1200
S=2
T=1000
L=8
N=5
z=1

if z == 1:
    active_circuits = torch.tensor(range(T), device=device).reshape(T,1)
elif z == 2:
    active_circuits = torch.randint(T, (bs, z), device=device)
    same = active_circuits[:,0] == active_circuits[:,1]
    if same.any():
        active_circuits[same, 1] = (active_circuits[same, 0] + 1) % T
else:
    raise ValueError("z must be 1 or 2")

for n in range(N):
    for d in [3,4]:
        for S in [3,5]:
        
            if d == 4:
                net = RotInSupNetwork_4d(D/4,T,S,device=device)
                if S == 5:
                    color = 'green'
                else:
                    color = 'blue'
            else:
                net = RotInSupNetwork_3d(D/3,T,S,device=device)
                if S == 5:
                    color = 'orange'
                else:
                    color = 'red'

            run = net.run(L, active_circuits=active_circuits)

            error = (((run.x - run.est_x)**2).sum(-1)**0.5)[:,:,0]
            max_error, _ = error.max(1)

            if n == 0:
                plt.plot(max_error.cpu().numpy(), color=color, marker='o', markersize=2, alpha=0.5,
                        label = f'd={d}, S={S}')
            else:
                plt.plot(max_error.cpu().numpy(), color=color, marker='o', markersize=2, alpha=0.5)

plt.grid()
plt.xlabel('Layer')
plt.ylabel('Max Error')
plt.title(f'D={D}, T={T}, z={z}')
plt.legend()
plt.xticks(range(L))
y_min, y_max = plt.ylim()
plt.yticks(range(int(y_max)))
plt.grid(True)
plt.show()
# %%




# Ds = [1200] 

# Ts = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 
#       2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 
#       3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 
#       4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900,
#       5000]


Ds = [800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 
      2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]
Ts = [2000]

Ss = [3]
L = 5
bs = 1000
zs = [1, 2, 3, 4, 5]


mse_df = None
settings_df = None


for D in tqdm.tqdm(Ds):
    for T in Ts :
        for S in Ss:
            for z in zs:
            
                try:
                    test_net = RotInSupNetwork_4d(D/4,T,S,device=device)
                except MaxT:
                    continue

                test_run = test_net.run(L,z,bs,ideal=True)
                e = test_run.x - test_run.est_x
                mse = (e ** 2).mean((1,2)).sum((-1,))

                temp_mse_db = pd.DataFrame({
                    'Layer': range(0,L),
                    'MSE': mse.cpu().numpy(),
                    'z': z,
                    'D': D,
                    'S': S,
                    'T': T})
                
                temp_settings_db = pd.DataFrame({
                    'D': D,
                    'S': S,
                    'T': T,
                    'z': z,}, index=[0])

                settings_df = pd.concat([settings_df, temp_settings_db], ignore_index=True)

                mse_df = pd.concat([mse_df, temp_mse_db], ignore_index=True)



# %%
settings_other_than_T = settings_df.drop(columns=['T'])
settings_other_than_T = settings_other_than_T.drop_duplicates(ignore_index=True)


for setting in settings_other_than_T.itertuples():

    matching_rows = mse_df[
        (mse_df['D'] == setting.D) & 
        (mse_df['S'] == setting.S) & 
        (mse_df['z'] == setting.z)]

    if len(matching_rows) <= 2*L:
        continue

    for l in range(L):
        layer_db = matching_rows[matching_rows['Layer'] == l]
        plt.plot(layer_db['T'], layer_db['MSE'],  label=f'Layer {l}', marker='o')
    plt.xlabel('T')
    plt.ylabel('MSE')
    plt.title(f'D={setting.D}, S={setting.S}, z={setting.z}')
    plt.legend()
    plt.show()
# %%
settings_other_than_D = settings_df.drop(columns=['D'])
settings_other_than_D = settings_other_than_D.drop_duplicates(ignore_index=True)

for setting in settings_other_than_D.itertuples():

    matching_rows = mse_df[
        (mse_df['T'] == setting.T) & 
        (mse_df['S'] == setting.S) & 
        (mse_df['z'] == setting.z)]

    if len(matching_rows) <= 2*L:
        continue

    for l in range(L):
        layer_db = matching_rows[matching_rows['Layer'] == l]
        plt.plot([(4/D)**2 for D in layer_db['D']], layer_db['MSE'],  label=f'Layer {l}', marker='o')
    plt.xlabel('(d/D)^2')
    plt.ylabel('MSE')
    plt.title(f'T={setting.T}, S={setting.S}, z={setting.z}')
    plt.legend()
    plt.show()
# %%
