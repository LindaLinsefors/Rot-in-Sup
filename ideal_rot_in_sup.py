#%%

from bleach import clean
import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm

from networks import RunData

device = 'cpu' 
torch.set_default_device(device)

#Make sure networks.py and assignments.py are reloaded
import importlib, assignments
importlib.reload(assignments)

from assignments import (maxT, MaxT,
                         comp_in_sup_assignment,
                         expected_overlap_error, 
                         expected_squared_overlap_error, 
                         probability_of_overlap,
                         frequency_of_overlap,
                         Test)


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

#%% Plot
#   Plot


D = 1200
T = 1000

S = 5
z = 1
bs = 800
L = 4
Dod = D // 3

correction = 0

circ = RotSmallCircuits(T, 0)
net = IdealRotInSup(Dod, L, S, circ, correction)
run = net.run(z, bs)

mse = (run.est_x - run.x).pow(2).mean(dim=(1, 2)).sum(dim=-1)
no_mask_mse = (run.no_msk_est_x - run.x).pow(2).mean(dim=(1, 2)).sum(dim=-1)

plt.plot(mse.cpu().numpy(), marker='o', label='MSE')
plt.plot(no_mask_mse.cpu().numpy(), marker='o', label='No Mask MSE')
plt.plot([expected_mse(T,Dod,l,0)[1] for l in range(L+1)], linestyle='--', marker='x', label='Expected MSE')
plt.grid(True)
plt.xlabel('Layer')
plt.xticks(range(L+1))
plt.legend()

plt.title(f'Dod={Dod}, T={T}, S={S}, z={z}, bs={bs}')

plt.show()









#%% 


from ipywidgets import interact
def f(x, y):
    return x+y
interact(f, x=10, y=20)
# %%
