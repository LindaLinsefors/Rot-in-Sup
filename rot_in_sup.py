#%% Setup
#   Setup

import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm
from assignments import maxT, comp_in_sup_assignment

#Make sure networks.py is reloaded
import importlib, networks
importlib.reload(networks)

from networks import RotInSupNetwork_4d as RotInSupNetwork
from networks import SmallCircuits

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

smal_test_net = RotInSupNetwork(Dod,T,S)
test_run = smal_test_net.run(L,z,bs)

for k in range(z):
    for b in range(bs):
        print(f'\n circuit {k} in batch {b}\n')
        for l in range(L):
            print(f'l={l}')
            print('x:    ', test_run.x[l,b,k])
            print('est_x:', test_run.est_x[l,b,k], '\n')


#%% Larger test
#   Larger test
Dod=600
S=5
T=4000
L=5
z=2
bs=2

test_net = RotInSupNetwork(Dod,T,S)
test_run = test_net.run(L,z,bs)

for k in range(z):
    for b in range(bs):
        print(f'\n circuit {k} in batch {b}\n')
        for l in range(L):
            print(f'l={l}')
            print('x:    ', test_run.x[l,b,k])
            print('est_x:', test_run.est_x[l,b,k], '\n')

# %%
Dod=600
D=int(4*Dod)
S=5
T=4000
L=8
N = 3600

net = RotInSupNetwork(Dod,T,S)
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
    netS[S] = RotInSupNetwork(Dod,T,S)
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
Dod=5
S=2
T=2
L=3
z=1
bs = 2
device='cpu'
L=2
balance = True

#Small circuits
small_circuits = SmallCircuits(T, device)
r = small_circuits.r

#Function parameters
Dod = int(Dod) # Number of neurons in the large network divided by 4
T = int(T) # Number of small circuits in superposition
S = int(S) # Number of large network neurons used by each small circuit neuron
L = int(L) # Number of layers before W repeats

#Embedding assignments for the on indicator
assignments_on = torch.randn(T, Dod, device=device)
assignments_on = assignments_on / assignments_on.norm(dim=1, keepdim=True)

#Embedding assignments for the vector values
assignments = torch.zeros(L, T, Dod)

assignments[0], _ = comp_in_sup_assignment(T, Dod, S, device)

for l in range(1,L):
    shuffle = torch.randperm(T, device=device)
    assignments[l], _ = assignments[0,shuffle]

#Used for corelated computations only
if balance:
    #Slightly negative for non assigned neurons s.t. balanced_assignments.mean()=0
    balanced_assignments = assignments * (1 + S/(Dod - S)) - torch.ones_like(assignments) * S/(Dod - S) 
else:
    balanced_assignments = assignments

#Used as index to get assigment from previous layer
previous = torch.roll(torch.arange(0, T, device=device), shifts=1)

#One vector
one = torch.ones(2, device=device)

#Large network weight matrices
W = torch.zeros(L, 3*Dod, 3*Dod, device=device)


#Preserving activation indicators
W[:, :Dod, :Dod] = torch.eye(Dod, device=device)[None, :, :]

#Adding 2 to active circuit neurons and suptracting rotation of one-vector
W[:, Dod:2*Dod, :Dod], W[:, 2*Dod:, :Dod] = (
            2 * torch.einsum('ltn,tm,i->ilnm', (balanced_assignments, assignments_on, one)) 
            - torch.einsum('ltn,tm,tij,j->ilnm', (assignments, assignments_on, r, one)) )

#Rotating
(W[:, Dod:2*Dod, Dod:2*Dod], W[:, Dod:2*Dod, 2*Dod:]), 
(W[:, 2*Dod:,    Dod:2*Dod], W[:, 2*Dod:,    2*Dod:]) = (
            torch.einsum('ltn,ltm,tij->ijnm', (assignments, assignments[previous], r)) / S)


# %%
L_W = L #Number of layers before W repeats

L = 5
z = 2
bs = 1
run_name = None

#Function parameters
L = int(L) # Number of layers
z = int(z) # Number of circuits in superposition
bs = int(bs) # Batch size

#Import network data as local variables
#Fill in later

#Run small circuits
x, active_circuits = small_circuits.run(L, z, bs)

#Large network initial values
A = torch.zeros(L, bs, 3*Dod, device=device)
A[0, :, :Dod] = torch.einsum('bti->bi', assignments_on[active_circuits])
A[0, :, Dod:2*Dod], A[0, :, Dod*2:] = torch.einsum('btn,bti->ibn', (assignments[1, active_circuits], x[1]))

est_x = torch.zeros(L, bs, z, 2, device=device)
est_x[0] = x[0]

#Running the large network: Layer 1
A[1] = torch.relu(A[0] + 1) - 1 #Coppying over everything from previous layer

est_x[1,:,:,0] = torch.einsum('btn,bn->bt', (assignments[1, active_circuits], A[1, :, Dod:2*Dod])) / S
est_x[1,:,:,1] = torch.einsum('btn,bn->bt', (assignments[1, active_circuits], A[1, :, Dod*2:])) / S

#Running the large network: Layer 2
A[2, :, :Dod] = A[1, :, :Dod] #Coppying over activation values from previous layer
A[2, :, Dod:] = torch.relu(torch.einsum('bnm,bm->bn', (W[2%L_W, Dod:, Dod:], A[1, :, Dod:]))  #Rotating 
                 + 2 * torch.tile(torch.einsum('tn,tm,bm->bn', (balanced_assignments[2], assignments_on, A[1, :, :Dod])), 
                                  dims = (2)) #Add 2 to active
                 - 1) #bias = -1
                             
#All other layers
for l in range(3,L):
    A[l] = torch.einsum('nm,bm->n' (W[l%L_W], A[l-1]))
    A[l, :, Dod:] = torch.relu(A[l, :, Dod:] - 1)


for l in range(2,L):
    est_x[l,:,:,0] = torch.einsum('btn,bn->bt', (assignments[l, active_circuits], A[l, :, Dod:2*Dod])) / S - 1
    est_x[l,:,:,1] = torch.einsum('btn,bn->bt', (assignments[l, active_circuits], A[l, :, Dod*2:])) / S - 1
    

#%%

v = torch.randint(10, (2, 3, 4))
print(v)
fv = v.flatten(start_dim=1)
print(fv)
# %%

torch.tile()