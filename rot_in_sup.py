#%% Setup
#   Setup

import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm

#Make sure networks.py and assignments.py are reloaded
import importlib, networks, assignments
importlib.reload(networks)
importlib.reload(assignments)

from networks import RotInSupNetwork_4d, RotInSupNetwork_3d
from assignments import maxT


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


#%% Small test
#   Small test
Dod=5
S=2
T=2
L=4
z=1
bs = 2

smal_test_net = RotInSupNetwork_4d(Dod,T,S)
test_run = smal_test_net.run(L,z,bs)

for k in range(z):
    for b in range(bs):
        print(f'\n circuit {k} in batch {b}\n')
        for l in range(L):
            print(f'l={l}')
            print('x:    ', test_run.x[l,b,k])
            print('est_x:', test_run.est_x[l,b,k], '\n')


smal_test_net = RotInSupNetwork_3d(Dod,T,S)
test_run = smal_test_net.run(L,z,bs)

for k in range(z):
    for b in range(bs):
        print(f'\n circuit {k} in batch {b}\n')
        for l in range(L):
            print(f'l={l}')
            print('x:    ', test_run.x[l,b,k])
            print('est_x:', test_run.est_x[l,b,k], '\n')


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

test_net = RotInSupNetwork_3d(D/3,T,S,L,balance=True)
test_run = test_net.run(L,z,bs)
e = test_run.x - test_run.est_x
mse = (e ** 2).mean((1,2)).sum((-1,))
ste = mse**0.5
mse_results.append(mse)
ste_results.append(ste)
labels.append('d=3 Network, balance=True')

test_net = RotInSupNetwork_3d(D/3,T,S,L,balance=False)
test_run = test_net.run(L,z,bs)
e = test_run.x - test_run.est_x
mse = (e ** 2).mean((1,2)).sum((-1,))
ste = mse**0.5
mse_results.append(mse)
ste_results.append(ste)
labels.append('d=3 Network, balance=False')

test_net = RotInSupNetwork_4d(D/4,T,S)
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

#%% Plotting the results
#   Plotting the results

# Plot MSE
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
for i, mse in enumerate(mse_results):
    plt.plot(mse, label=labels[i], marker='o')
plt.xlabel('Layer')
plt.ylabel('Mean Squared Error')
plt.title(title)
plt.legend()

# Plot STE
plt.subplot(1, 2, 2)
for i, ste in enumerate(ste_results):
    plt.plot(ste, label=labels[i], marker='o')
plt.xlabel('Layer')
plt.ylabel('Standard Error')
plt.title(title)
plt.legend()

plt.tight_layout()
plt.show()

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

# %%
