#%%

import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm
device = 'cpu' 

#Make sure networks.py and assignments.py are reloaded
import importlib, networks, assignments
importlib.reload(networks)
importlib.reload(assignments)

from assignments import (maxT, MaxT,
                         comp_in_sup_assignment,
                         expected_overlap_error, 
                         expected_squared_overlap_error, 
                         probability_of_overlap,
                         frequency_of_overlap)



#%% Set up
#   Set up


class RunData:
   pass

class RotSmallCircuits:
    def __init__(self, T, b, device=device):
        self.T = T # Number of small circuits
        self.d = 3 # Number of neurons per small circuit
        self.device = device
        
        #Small circuit rotations
        theta = torch.rand(T,device=device) * 2 * np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        
        self.w = torch.zeros(T, 3, 3, device=device)

        self.w[:,0,0] = 1 + b
        self.w[:,1,0] = 1 + b - cos + sin
        self.w[:,2,0] = 1 + b - cos - sin

        self.w[:,1,1] = cos
        self.w[:,1,2] = -sin
        self.w[:,2,1] = sin
        self.w[:,2,2] = cos

        self.b = - torch.ones(3, device=device) * b

    def run(self, L, z, bs, active_circuits=None):
        """Run all small circuits on input random inputs"""

        device = self.device
        a = torch.zeros(L+1, bs, z, 3, device=device)

        #Active circuits
        if active_circuits is None:  # Generating random circuits
            active_circuits = torch.randint(self.T, (bs, z), device=device)

            # Replace any duplicates with non-duplicates
            same = torch.zeros(bs, dtype=torch.bool, device=device)
            for i in range(z):
                for j in range(i):
                    same += (active_circuits[:,i] == active_circuits[:,j])
            n = same.sum()
            active_circuits[same] = torch.tensor(range(z*n), dtype=torch.int64, 
                                                 device=device).reshape(n, z) % self.T

        #Initial values
        initial_angle = torch.rand(bs, z, device=device) * 2 * np.pi
        a[0, :, :, 0] = 1
        a[0, :, :, 1] = torch.cos(initial_angle) + 1
        a[0, :, :, 2] = torch.sin(initial_angle) + 1

        #Running the small circuits
        for l in range(L):
            a[l+1] = torch.relu(
                torch.einsum('btij,btj->bti', self.w[active_circuits], a[l]) 
                + self.b)

        return a, active_circuits

class CompInSup:
    def __init__(self, D, L, S, small_circuits, correction=None, device=device):
        self.device = device

        d = small_circuits.d  # Number of neurons in each small circuit
        T = small_circuits.T
        D = int(D)
        Dod = int(D/d)
        w = small_circuits.w

        self.T = T # Number of small circuits
        self.D = D # Number of neurons in the large network
        self.L = L # Number of layers in the large network
        self.S = S # Number of large network neurons used by each small circuit neuron
        self.small_circuits = small_circuits
        self.correction = correction # Neggative correction for unembedding

        embed = torch.zeros(L, T, Dod, device=device)
        unemb = torch.zeros(L, T, Dod, device=device)
        assign = torch.zeros(L, T, S, device=device, dtype=torch.int64)

        embed[0], assign[0] = comp_in_sup_assignment(T, Dod, S, device)
        for l in range(1, L):
            shuffle = torch.randperm(T, device=device)
            embed[l] = embed[0][shuffle]
            assign[l] = assign[0][shuffle]
        
        if correction is None:
            p = probability_of_overlap(T, Dod, S)
            correction = p/((S-p)*S)
        unemb = - torch.ones(L, T, Dod, device=device) * correction
        unemb += embed * (1/S + correction)

        W = torch.zeros(L, D, D, device=device)
        W[0] = torch.eye(D, device=device)
        for l in range(1,L):
            [[W[l,:Dod,     :Dod], W[l,:Dod,     Dod:2*Dod], W[l,:Dod,     2*Dod:]],
             [W[l,Dod:2*Dod,:Dod], W[l,Dod:2*Dod,Dod:2*Dod], W[l,Dod:2*Dod,2*Dod:]],
             [W[l,2*Dod:,   :Dod], W[l,2*Dod:,   Dod:2*Dod], W[l,2*Dod:,   2*Dod:]]
            ]= torch.einsum('tn,tij,tm->ijnm', unemb[l], w, embed[l-1])

        #First bias is zero, the rest are the biases of the small circuits
        B = torch.zeros(L, D, device=device)
        B[1:, :Dod] = small_circuits.b[0]
        B[1:, Dod:2*Dod] = small_circuits.b[1]
        B[1:, 2*Dod:] = small_circuits.b[2]

        self.W = W
        self.B = B
        self.embed = embed
        self.unemb = unemb
        self.assign = assign

    def run(self, L, z, bs, active_circuits=None):
        device = self.device

        a, active_circuits = self.small_circuits.run(L, z, bs, active_circuits)
        x = a[:,:,:,1:] - 1

        Dod = self.D // 3
        
        A = torch.zeros(L+1, bs, self.D, device=device)
        [A[0,:,:Dod], A[0,:,Dod:2*Dod], A[0,:,2*Dod:]] = torch.einsum('btn,bti->ibn', self.embed[0,active_circuits],a[1])
        for l in range(L):
            A[l+1] = torch.einsum('nm,bm->bn', self.W[l], A[l]) + self.B[l]
            A[l+1] = torch.relu(A[l+1])

        est_a = torch.zeros(L+1, bs, z, 3, device=device)
        est_a[0] = a[0]
        for l in range(L):
            est_a[l+1, :, :, 0] = torch.einsum('btn,bn->bt', self.unemb[l, active_circuits], A[l+1,:,:Dod     ])
            est_a[l+1, :, :, 1] = torch.einsum('btn,bn->bt', self.unemb[l, active_circuits], A[l+1,:,Dod:2*Dod])
            est_a[l+1, :, :, 2] = torch.einsum('btn,bn->bt', self.unemb[l, active_circuits], A[l+1,:,2*Dod:   ])

        est_x = est_a[:, :, :, 1:] - est_a[:, :, :, 0][:, :, :, None]

        run = RunData()
        run.a = a
        run.x = x
        run.est_a = est_a
        run.est_x = est_x
        run.active_circuits = active_circuits
        run.A = A
        return run
# %% Very small test
#    Very small test

Dod=5
D=Dod*3
S=2
T=2
L=4
z=1
bs=1

circ = RotSmallCircuits(T, 0.1, device=device)
net = CompInSup(D, L, S, circ, correction=0, device=device)
run = net.run(L, z, bs, active_circuits=torch.tensor([[0]], device=device))

if (run.x - run.est_x).sum().abs() > 1e-6 and (run.a - run.est_a).sum().abs() > 1e-6:
    print("CompInSup test failed: The output does not match the expected result.")
else:
    print("CompInSup test passed: The output matches the expected result.")



#%% 

Dod = 100
D = 300
T = 120
S = 5
z = 1
bs = 800
L = 10

p = probability_of_overlap(T, Dod, S)
f = frequency_of_overlap(T, Dod, S)

print(f"Probability of overlap: {p:.4f}")
print(f"Frequency of overlap: {f:.4f}")
#%%
correction = f/((S-f)*S)
#correction = 1/Dod

circ = RotSmallCircuits(T, 0.1, device=device)
net = CompInSup(D, L, S, circ, device=device)
run = net.run(L, z, bs)

x = run.x
est_x = run.est_x
a = run.a
est_a = run.est_a

print(est_a[:,:,0,0].mean((-1)))

# %%

D = 1200
T = 1000

# D = 300
# T = 120

S = 5
z = 1
bs = 800
L = 7
Dod = D // 3
b = 1

p = probability_of_overlap(T, Dod, S)
f = frequency_of_overlap(T, Dod, S)

print(f"Probability of overlap: {p:.4f}")
correction = p/((S-p)*S)
print(f"Correction: {correction:.4f}")

circ = RotSmallCircuits(T, b, device=device)
net = CompInSup(D, L, S, circ, correction=correction, device=device)
run = net.run(L, z, bs)
print(run.est_a[:,:,0,0].mean((-1)))

print(f"Frequency of overlap: {f:.4f}")
correction = f/((S-f)*S)
print(f"Correction: {correction:.4f}")

circ = RotSmallCircuits(T, b, device=device)
net = CompInSup(D, L, S, circ, correction=correction, device=device)
run = net.run(L, z, bs)
print(run.est_a[:,:,0,0].mean((-1)))

print("Correction = 1/(Dod-S)")
correction = 1/(Dod-S)
print(f"Correction: {correction:.4f}")
circ = RotSmallCircuits(T, b, device=device)
net = CompInSup(D, L, S, circ, correction=correction, device=device)
run = net.run(L, z, bs)
print(run.est_a[:,:,0,0].mean((-1)))

# %%

Dod = 100
D = 300
T = 120
S = 5

p = probability_of_overlap(T, Dod, S)
f = frequency_of_overlap(T, Dod, S)

corr_p = p/((S-p)*S)
corr_f = f/((S-f)*S)

embed, assign = comp_in_sup_assignment(T, Dod, S, device)

unemb_p = - torch.ones(T, Dod, device=device) * corr_p
unemb_p += embed * (1/S + corr_p)

unemb_f = - torch.ones(T, Dod, device=device) * corr_f
unemb_f += embed * (1/S + corr_f)

E_f = (unemb_f @ embed.T - torch.eye(T)).sum()/(T * (T - 1))
E_p = (unemb_p @ embed.T - torch.eye(T)).sum()/(T * (T - 1))
# %%

Dod = 100
D = 300
T = 120
S = 5

t=0
mask = torch.ones(T, dtype=torch.bool)
mask[t] = False

neighbour_neurons = torch.where((embed[mask].T @ embed[mask] @ embed[t])>0.5, 1, 0) - embed[t]
unemb_t = (embed[t] - 1/(S) * neighbour_neurons)/S



# %%
