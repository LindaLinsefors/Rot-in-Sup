# %% Setup step_in_sup
#    Setup step_in_sup


import numpy as np
import torch
import matplotlib.pyplot as plt

import importlib, classes_and_functions, assignments
importlib.reload(classes_and_functions)
importlib.reload(assignments)

from assignments import maxT

from classes_and_functions import (CompInSup, 
                                   random_active_circuits,
                                   get_inactive_circuits)

class StepSmallCircuits():
    def __init__(self, T):

        d = 2
        T = int(T)

        self.T = T
        self.d = d

        self.mean_w = torch.tensor([[2.0, -2.0],
                                    [2.0, -2.0]])
            
        self.b = torch.tensor([-0.5, 
                               -1.5])
        
        self.diff_w = torch.zeros(T, d, d)

        self.w = self.mean_w[None, :, :] + self.diff_w

    def run(self, L, z, bs, active_circuits=None):

        d = self.d
        T = self.T

        if active_circuits is None:
            active_circuits = random_active_circuits(T, bs, z)

        a = torch.zeros(L+1, bs, z, d)

        a[0, :, :, 0] = 1.5
        a[0, :, :, 1] = 0.5

        for l in range(L):
            a[l+1] = torch.relu(
                torch.einsum('btij,btj->bti', self.w[active_circuits], a[l]) 
                + self.b)

        return a, active_circuits
    

def expected_mse_step(T, Dod, l, z):
    if l==0:
        return 0
    
    return 
    












#%%

T = 10
L = 5
Z = 3
BS = 2

circ = StepSmallCircuits(T)
a, active_circuits = circ.run(L, Z, BS)

a

# %%

T = 1000

D = 1200
d = 4
Dod = D // d

L = 7
bs = 800
n = 1

S = 3
z = 1

correction = 0

circ = StepSmallCircuits(T)

for correction in [None]:
    #for z in [3,2,1]:
        mse = torch.zeros(n, L+1)
        for i in range(n):
            net = CompInSup(D, L, S, circ, correction=correction)
            run = net.run(L, z, bs)
            a = 1
            est_a = run.est_a[:,:,:,0] - run.est_a[:,:,:,1]
            mse[i] = (est_a - 1).pow(2).mean(dim=(1, 2))

            if correction == 0:
                label = f'no correction'
            else:
                label = f' '

            if i == 0:
                line = plt.plot(mse[i], label=label, marker='o')
            else:
                plt.plot(mse[i], marker='o', color=line[0].get_color())

plt.legend()
plt.grid(True)
plt.xlabel('Layer')
plt.ylabel('MSE')
plt.title(f'T={T}, D={D}, Dod={Dod}, S={S}, z={z}\nbs={bs}, n={n}')
plt.show()
# %% Run and plot active circuits

T = 1000

D = 600
d = 2
Dod = D // d

L = 4
bs = 800
n = 1

S = 5
z = 1

correction = None

circ = StepSmallCircuits(T)


plt.figure(figsize=(10, 5))

for S in [3,5]:
    for capped, correction in [(False, 0)]:
        #for z in [2,1]:

            if capped:
                label = f'z={z}, S={S}, capped'
            elif correction is None:
                label = f'z={z}, S={S}'
            elif correction == 0:
                label = f'z={z}, S={S}, no correction'
            else:
                label = '?'

            for i in range(n):
                net = CompInSup(D, L, S, circ, correction=correction)
                run = net.run(L, z, bs, capped=capped)

                mse = (run.est_a - run.a).pow(2).mean(dim=(1, 2))

                plt.subplot(1, 3, 1)
                if i == 0:
                    line = plt.plot(mse[:, 0], label=label, marker='o')
                else:
                    plt.plot(mse[:, 0], marker='o', color=line[0].get_color())

                plt.subplot(1, 3, 2)
                plt.plot(mse[:, 1], marker='o', color=line[0].get_color())  

                est_a = run.est_a[:,:,:,0] - run.est_a[:,:,:,1]
                mse = (est_a - 1).pow(2).mean(dim=(1, 2))
                plt.subplot(1, 3, 3)
                plt.plot(mse, marker='o', color=line[0].get_color())

for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.grid(True)
    plt.xlabel('Layer')
    plt.ylabel('MSE')
    if i == 0:
        plt.title('a_1')
        plt.legend()
    elif i == 1:
        plt.title('a_2')
    elif i == 2:
        plt.title('a_1 - a_2')
plt.suptitle(f'T={T}, D={D}, D/d={Dod}, z={z}\nbs={bs}, n={n}')
plt.tight_layout()
plt.show()
#%%


























# %% Run Inactive
#    Run Inactive
T = 1000

D = 600
d = 2
Dod = D // d

L = 3
bs = 800
n = 1


plt.figure(figsize=(10, 5))
for z in [2,1]:
    for S in [5,4,3]:
        for capped, correction in [(False, None), (True, None)]:

            circ = StepSmallCircuits(T)
            net = CompInSup(D, L, S, circ, correction=correction)
            run = net.run(L, z, bs, capped=capped)

            unemb = net.unemb
            A = run.A
            active_circuits = run.active_circuits
            inactive_circuits = get_inactive_circuits(active_circuits, T)

            inactive_est_a = torch.zeros(L+1, bs, T-z, d)
            for l in range(L):
                for i in range(d):
                    inactive_est_a[l+1, :, :, i] = torch.einsum('btn,bn->bt', unemb[l, inactive_circuits], A[l+1,:,i*Dod:(i+1)*Dod])



            mse = inactive_est_a.pow(2).mean(dim=(1, 2))
            mse_diff = (inactive_est_a[:,:,:,0] - inactive_est_a[:,:,:,1]).pow(2).mean(dim=(1, 2))

            plt.subplot(1, 3, 1)
            label = f'z={z}, S={S}, {("capped" if capped else ("no correction" if correction == 0 else ""))}'
            line = plt.plot(mse[:, 0], label=label, marker='o')

            plt.subplot(1, 3, 2)
            plt.plot(mse[:, 1], marker='o', color=line[0].get_color())  

            plt.subplot(1, 3, 3)
            plt.plot(mse_diff, marker='o', color=line[0].get_color())

for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.grid(True)
    plt.xlabel('Layer')
    plt.ylabel('MSE')
    if i == 0:
        plt.title('a_1')
        plt.legend()
    elif i == 1:
        plt.title('a_2')
    elif i == 2:
        plt.title('a_1 - a_2')
plt.suptitle(f'MSE for Inactive Circuits\nT={T}, D={D}, D/d={Dod}, bs={bs}')
plt.tight_layout()
plt.show()



# %% Plot Inactive
#    Plot Inactive











bs = 1


circ = StepSmallCircuits(T)
net = CompInSup(D, L, S, circ)
run = net.run(L, z, bs, capped=capped)

unemb = net.unemb
A = run.A
active_circuits = run.active_circuits
inactive_circuits = get_inactive_circuits(active_circuits, T)

inactive_est_a = torch.zeros(L+1, bs, T-z, d)
for l in range(L):
    for i in range(d):
        inactive_est_a[l+1, :, :, i] = torch.einsum('btn,bn->bt', unemb[l, inactive_circuits], A[l+1,:,i*Dod:(i+1)*Dod])

n = 10


plt.figure(figsize=(10, 5))

for i in range(n):
    plt.subplot(1, 3, 1)
    line = plt.plot(inactive_est_a[:, 0, i, 0], marker='o')

    plt.subplot(1, 3, 2)
    line = plt.plot(inactive_est_a[:, 0, i, 1], marker='o', color=line[0].get_color())

    plt.subplot(1, 3, 3)
    plt.plot(inactive_est_a[:, 0, i, 0] - inactive_est_a[:, 0, i, 1], marker='o', color=line[0].get_color())

for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.grid(True)
    plt.xlabel('Layer')
    plt.ylabel('MSE')
    if i == 0:
        plt.title('a_1')
    elif i == 1:
        plt.title('a_2')
    elif i == 2:
        plt.title('a_1 - a_2')
plt.suptitle(f'T={T}, D={D}, D/d={Dod}\nbs={bs}, n={n}')
plt.tight_layout()
plt.show()
# %%
