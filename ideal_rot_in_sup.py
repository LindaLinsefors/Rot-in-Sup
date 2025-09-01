#%% Setup
#   Setup

from bleach import clean
import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm
import numpy as np
from scipy.stats import norm


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
                         frequency_of_overlap,
                         Test)

from classes_and_functions import (RunData, 
                                   RotSmallCircuits, 
                                   expected_mse_rot, 
                                   get_inactive_circuits)


# Fake import of RunData
# Fake import of RotSmallCircuits

class IdealRotInSup:
    def __init__(self, Dod, L, S, small_circuits, correction=None):
        self.Dod = Dod
        self.L = L
        self.S = S
        self.small_circuits = small_circuits
        self.correction = correction
        self.T = small_circuits.T
        self.r = small_circuits.r

        T = self.T
        r = self.r

        embed = torch.zeros(L, T, Dod)
        unemb = torch.zeros(L, T, Dod)
        assign = torch.zeros(L, T, S, dtype=torch.int64)

        embed[0], assign[0] = comp_in_sup_assignment(T, Dod, S)
        for l in range(1, L):
            shuffle = torch.randperm(T)
            embed[l] = embed[0][shuffle]
            assign[l] = assign[0][shuffle]

        unemb = - torch.ones(L, T, Dod) * correction
        unemb += embed * (1/S + correction)

        R = torch.zeros(L, 2*Dod, 2*Dod)
        R[0] = torch.eye(2*Dod)

        for l in range(1,L):
            [[R[l,:Dod, :Dod], R[l,:Dod, Dod:]],
             [R[l, Dod:, :Dod], R[l,Dod:, Dod:]]
            ]= torch.einsum('tn,tij,tm->ijnm', unemb[l], r, embed[l-1])


        self.embed = embed
        self.unemb = unemb
        self.assign = assign

        self.R = R

    def run(self, z, bs):

        Dod = self.Dod
        L = self.L
        T = self.T
        S = self.S
        embed = self.embed
        R = self.R

        a, active_circuits = self.small_circuits.run(L, z, bs)
        x = a[:, :, :, 1:] - 1

        X = torch.zeros(L+1, bs, 2*Dod)
        pre_X = torch.zeros(L+1, bs, 2*Dod)
        no_msk_X = torch.zeros(L+1, bs, 2*Dod)

        [X[0,:,:Dod], X[0,:,Dod:]] = torch.einsum('btn,bti->ibn', embed[0, active_circuits], x[1])
        no_msk_X[0] = X[0]
        pre_X[0] = X[0]

        for l in range(L):
            pre_X[l+1] = torch.einsum('nm,bm->bn', R[l], X[l])
            no_msk_X[l+1] = torch.einsum('nm,bm->bn', R[l], no_msk_X[l])

            mask = torch.einsum('btn->bn', embed[l, active_circuits]) != 0
            X[l+1, :, Dod:][mask] = pre_X[l+1, :, Dod:][mask]
            X[l+1, :, :Dod][mask] = pre_X[l+1, :, :Dod][mask]


        est_x = torch.zeros(L+1, bs, z, 2)
        no_msk_est_x = torch.zeros(L+1, bs, z, 2)

        est_x[0] = x[0]
        no_msk_est_x[0] = x[0]

        for l in range(L):
            est_x[l+1, :, :, 0] = torch.einsum('btn,bn->bt', self.unemb[l, active_circuits], X[l+1, :, :Dod])
            est_x[l+1, :, :, 1] = torch.einsum('btn,bn->bt', self.unemb[l, active_circuits], X[l+1, :, Dod:])

            no_msk_est_x[l+1, :, :, 0] = torch.einsum('btn,bn->bt', self.unemb[l, active_circuits], no_msk_X[l+1, :, :Dod])
            no_msk_est_x[l+1, :, :, 1] = torch.einsum('btn,bn->bt', self.unemb[l, active_circuits], no_msk_X[l+1, :, Dod:])

        run = RunData()

        run.X = X
        run.pre_X = pre_X
        run.no_msk_X = no_msk_X
        run.est_x = est_x
        run.no_msk_est_x = no_msk_est_x

        run.active_circuits = active_circuits
        run.x = x

        return run
    

class IdealCompInSup:
    def __init__(self, Dod, L, S, small_circuits, correction=None):
        self.Dod = Dod
        self.L = L
        self.S = S
        self.small_circuits = small_circuits
        self.correction = correction
        self.T = small_circuits.T
        self.w = small_circuits.w
        self.d = small_circuits.d
        self.rot = small_circuits.rot

        T = self.T
        w = self.w
        d = self.d

        embed = torch.zeros(L, T, Dod)
        unemb = torch.zeros(L, T, Dod)
        assign = torch.zeros(L, T, S, dtype=torch.int64)

        embed[0], assign[0] = comp_in_sup_assignment(T, Dod, S)
        for l in range(1, L):
            shuffle = torch.randperm(T)
            embed[l] = embed[0][shuffle]
            assign[l] = assign[0][shuffle]

        unemb = - torch.ones(L, T, Dod) * correction
        unemb += embed * (1/S + correction)

        W = torch.zeros(L, d*Dod, d*Dod)
        W[0] = torch.eye(d*Dod)

        if d == 3:
            for l in range(1,L):
                [[W[l,:Dod,     :Dod], W[l,:Dod,     Dod:2*Dod], W[l,:Dod,     2*Dod:]],
                 [W[l,Dod:2*Dod,:Dod], W[l,Dod:2*Dod,Dod:2*Dod], W[l,Dod:2*Dod,2*Dod:]],
                 [W[l,2*Dod:,   :Dod], W[l,2*Dod:,   Dod:2*Dod], W[l,2*Dod:,   2*Dod:]]
                ]= torch.einsum('tn,tij,tm->ijnm', unemb[l], w, embed[l-1])
        if d == 2:
            for l in range(1,L):
                [[W[l, :Dod, :Dod], W[l, :Dod, Dod:]],
                 [W[l, Dod:, :Dod], W[l, Dod:, Dod:]]
                ]= torch.einsum('tn,tij,tm->ijnm', unemb[l], w, embed[l-1])


        B = torch.zeros(L, D, device=device)
        for i in range(d):
            B[1:, i*Dod:(i+1)*Dod] = small_circuits.b[i]

        self.embed = embed
        self.unemb = unemb
        self.assign = assign

        self.W = W
        self.B = B

    def run(self, z, bs):

        Dod = self.Dod
        L = self.L
        T = self.T
        S = self.S
        embed = self.embed
        W = self.W
        B = self.B
        d = self.d
        rot = self.rot

        a, active_circuits = self.small_circuits.run(L, z, bs)
        if rot:
            x = a[:, :, :, 1:] - 1

        A = torch.zeros(L+1, bs, d*Dod)
        pre_A = torch.zeros(L+1, bs, d*Dod)
        no_msk_A = torch.zeros(L+1, bs, d*Dod)

        [A[0,:,:Dod], A[0,:,Dod:2*Dod], A[0,:,2*Dod:]] = torch.einsum('btn,bti->ibn', embed[0, active_circuits], a[1])
        no_msk_A[0] = A[0]
        pre_A[0] = A[0]

        for l in range(L):
            pre_A[l+1] = torch.einsum('nm,bm->bn', W[l], A[l]) + B[l]
            no_msk_A[l+1] = torch.einsum('nm,bm->bn', W[l], no_msk_A[l]) + B[l]

            mask = torch.einsum('btn->bn', embed[l, active_circuits]) != 0
            for i in range(d):
                A[l+1, :, i*Dod:(i+1)*Dod][mask] = pre_A[l+1, :, i*Dod:(i+1)*Dod][mask]

        est_a = torch.zeros(L+1, bs, z, d)
        no_msk_est_a = torch.zeros(L+1, bs, z, d)

        est_a[0] = a[0]
        no_msk_est_a[0] = a[0]

        for l in range(L):
            for i in range(d):
                est_a[       l+1, :, :, i] = torch.einsum('btn,bn->bt', self.unemb[l, active_circuits],        A[l+1, :, i*Dod:(i+1)*Dod])
                no_msk_est_a[l+1, :, :, i] = torch.einsum('btn,bn->bt', self.unemb[l, active_circuits], no_msk_A[l+1, :, i*Dod:(i+1)*Dod])

        if rot:
            est_x = torch.zeros(L+1, bs, z, 2)
            no_msk_est_x = torch.zeros(L+1, bs, z, 2)

            est_x[0] = x[0]
            no_msk_est_x[0] = x[0]

            est_x[1:, :, :, 0] = est_a[1:, :, :, 1] - est_a[1:, :, :, 0]
            est_x[1:, :, :, 1] = est_a[1:, :, :, 2] - est_a[1:, :, :, 0]

            no_msk_est_x[1:, :, :, 0] = no_msk_est_a[1:, :, :, 1] - no_msk_est_a[1:, :, :, 0]
            no_msk_est_x[1:, :, :, 1] = no_msk_est_a[1:, :, :, 2] - no_msk_est_a[1:, :, :, 0]

        run = RunData()

        run.active_circuits = active_circuits    

        run.A = A
        run.pre_A = pre_A
        run.no_msk_A = no_msk_A

        run.a = a
        run.est_a = est_a
        run.no_msk_est_a = no_msk_est_a

        if rot:
            run.x = x
            run.est_x = est_x
            run.no_msk_est_x = no_msk_est_x

        return run



#%% Plot
#   Plot


D = 1200
T = 1000

S = 5
z = 5
bs = 800
L = 3
Dod = D // 3
b = 0

circ = RotSmallCircuits(T, b)

version = 'Ideal Comp-in-Sup'
#version = 'Ideal Rot-in-Sup'

if version == 'Ideal Rot-in-Sup':
    NetClass = IdealRotInSup
    b = 0
if version == 'Ideal Comp-in-Sup':
    NetClass = IdealCompInSup

for z in [1, 2, 3, 4, 5]:
    #print(f'z={z}')

    # No special unembed
    correction = 0
    net = NetClass(Dod, L, S, circ, correction)
    run = net.run(z, bs)

    mse = (run.est_x - run.x).pow(2).mean(dim=(1, 2)).sum(dim=-1)
    no_mask_mse = (run.no_msk_est_x - run.x).pow(2).mean(dim=(1, 2)).sum(dim=-1)
    #print(f'observed MSE: {mse}')

    plt.plot(mse.cpu().numpy(), marker='o', label=f'z={z}, normal')
    plt.plot(no_mask_mse.cpu().numpy(), marker='o', label='z={z}, No Mask')

    # Special unembed
    correction = 1/(Dod-S)
    net = NetClass(Dod, L, S, circ, correction)
    run = net.run(z, bs)

    mse = (run.est_x - run.x).pow(2).mean(dim=(1, 2)).sum(dim=-1)
    no_mask_mse = (run.no_msk_est_x - run.x).pow(2).mean(dim=(1, 2)).sum(dim=-1)
    #print(f'expected MSE: {mse}\n')

    #plt.plot(mse.cpu().numpy(), marker='o', label=f'z={z}, Mod Un-embed')
    #plt.plot(no_mask_mse.cpu().numpy(), marker='o', label='No Mask & Mod Un-embed')

    # Expected error
    plt.plot([expected_mse_rot(T,Dod,l,b,z)[1] for l in range(L+1)], linestyle='--', marker='x', label=f'z={z}, Expected MSE')

# Other plot stuff
plt.grid(True)
plt.xlabel('Layer')
plt.xticks(range(L+1))
plt.legend()

plt.title(f'{version}\n Dod={Dod}, T={T}, S={S}, z={z}, bs={bs}')

plt.show()




#%%












D = 1200
T = 1000

S = 5
z = 1
bs = 800
L = 4
Dod = D // 3
b = 1

circ = RotSmallCircuits(T, b)

version = 'Ideal Comp-in-Sup'

# No special unembed
correction = 0
net = NetClass(Dod, L, S, circ, correction)
run = net.run(z, bs)

on = run.a[:, :, :, 0]
est_on = run.est_a[:, :, :, 0]
no_mask_est_on = run.no_msk_est_a[:, :, :, 0]

mse = (est_on - on).pow(2).mean(dim=(1, 2))
no_mask_mse = (no_mask_est_on - on).pow(2).mean(dim=(1, 2))

#plt.plot(mse.cpu().numpy(), marker='o', label='normal')
#plt.plot(no_mask_mse.cpu().numpy(), marker='o', label='No Mask')

# Special unembed
correction = 1/(Dod-S)
net = NetClass(Dod, L, S, circ, correction)
run = net.run(z, bs)

on = run.a[:, :, :, 0]
est_on = run.est_a[:, :, :, 0]
no_mask_est_on = run.no_msk_est_a[:, :, :, 0]

mse = (est_on - on).pow(2).mean(dim=(1, 2))
no_mask_mse = (no_mask_est_on - on).pow(2).mean(dim=(1, 2))

plt.plot(mse.cpu().numpy(), marker='o', label=f'z={z}, Mod Un-embed')
#plt.plot(no_mask_mse.cpu().numpy(), marker='o', label='No Mask & Mod Un-embed')

# Expected error
plt.plot([expected_mse_rot(T,Dod,l,b,z)[1] for l in range(L+1)], linestyle='--', marker='x', label=f'z={z}, Expected MSE')

# Other plot stuff
plt.grid(True)
plt.xlabel('Layer')
plt.xticks(range(L+1))
plt.legend()

plt.title(f'{version}\n Dod={Dod}, T={T}, S={S}, z={z}, bs={bs}, b={b}')

plt.show()
#%%



































#%% 


from ipywidgets import interact
def f(x, y):
    return x+y
interact(f, x=10, y=20)
# %%




















D = 800
T = 1000

S = 5
z = 2
bs = 800
L = 4
Dod = D // 2
b = 0
correction = 0


circ = RotSmallCircuits(T, b)
net = IdealRotInSup(Dod, L, S, circ, correction)
run = net.run(z, bs)


# %%

active_error = run.est_x - run.x
for l in [2,4]:

    mse = active_error[l].pow(2).mean().item()
    plt.hist(active_error[l].flatten().cpu().numpy(), bins=100, density=True, alpha=0.5, label=f'layer {l}, MSE={mse:.4f}')

    # Normal curve
    x = np.linspace(-0.4, 0.4, 200)
    variance = (expected_mse_rot(T, Dod, l, b, z)[1]/2)
    pdf = norm.pdf(x, loc=0, scale=variance**0.5)
    plt.plot(x, pdf, label=f'Normal, sigma^2={variance:.4f}')

plt.legend()
plt.title(f'Error distribution for active circuits\n D={D}, dT={2*T}, S={S}, z={z}, batch size={bs}')    
plt.show()
# %%

reduced_bs = bs

active = run.active_circuits[:reduced_bs]
inactive = get_inactive_circuits(active, T)
unemb = net.unemb
X = run.no_msk_X[:, :reduced_bs]

inactive_error = torch.zeros(L+1, reduced_bs, T-z, 2)

for l in range(L):
    inactive_error[l+1, :, :, 0] = torch.einsum('btn,bn->bt', unemb[l, inactive], X[l+1, :, :Dod]) 
    inactive_error[l+1, :, :, 1] = torch.einsum('btn,bn->bt', unemb[l, inactive], X[l+1, :, Dod:])

#%%

for l in [2,3]:

    mse = inactive_error[l].pow(2).mean().item()
    plt.hist(inactive_error[l].flatten().cpu().numpy(), bins=100, density=True, alpha=0.5, label=f'layer {l}, MSE={mse:.4f}')

    # Normal curve
    x = np.linspace(-0.5, 0.5, 200)
    variance = (expected_mse_rot(T, Dod, 1, b, z+1)[1]/2)
    pdf = norm.pdf(x, loc=0, scale=variance**0.5)
    plt.plot(x, pdf, label=f'Normal, sigma^2={variance:.4f}')

plt.legend()
plt.title(f'Error distribution for inactive circuits\n D={D}, dT={2*T}, S={S}, z={z}, batch size={bs}')    
plt.show()


for l in [2,3]:

    mse = inactive_error[l].pow(2).mean().item()
    plt.hist(inactive_error[l].flatten().cpu().numpy(), bins=100, density=True, alpha=0.5, label=f'layer {l}, MSE={mse:.4f}')

    # Normal curve
    x = np.linspace(-0.5, 0.5, 200)
    variance = (expected_mse_rot(T, Dod, 1, b, z+1)[1]/2)
    pdf = norm.pdf(x, loc=0, scale=variance**0.5)
    plt.plot(x, pdf, label=f'Normal, sigma^2={variance:.4f}')

plt.ylim(0, 0.5)
plt.legend()
plt.title(f'Error distribution for inactive circuits\n D={D}, dT={2*T}, S={S}, z={z}, batch size={reduced_bs}')    
plt.show()
# %%


for l in [2,4]:

    mse = inactive_error[l].pow(2).mean().item()
    plt.hist(inactive_error[l].flatten().cpu().numpy(), bins=100, density=True, alpha=0.5, label=f'layer {l}, MSE={mse:.4f}')

    # Normal curve
    x = np.linspace(-0.3, 0.3, 200)
    variance = (expected_mse_rot(T, Dod, 1, b, z+1)[1]/2)
    pdf = norm.pdf(x, loc=0, scale=variance**0.5)
    plt.plot(x, pdf, label=f'Normal, sigma^2={variance:.4f}')

plt.ylim(0, 0.0001)
plt.legend()
plt.title(f'Error distribution for inactive circuits\n D={D}, dT={2*T}, S={S}, z={z}, batch size={reduced_bs}')    
plt.show()
# %%
