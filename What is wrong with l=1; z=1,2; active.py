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
#%%

T = 500
D = 2*T
d = 4
Dod = D // d
S = 6

L = 3
z = 2

circ = RotSmallCircuits_4d(T, b=1)
net = CompInSup(D, L, S, circ)
run = net.run(L, z=z, bs=5, capped=True)
# %%


#print(run.x[1])
#print(run.est_x[1])
print(run.est_x[1]-run.x[1])

# %%
bs=T
run = net.run(L, z=z, bs=bs, capped=True)
(run.est_x[1]-run.x[1]).pow(2).sum(-1).mean()
# %%
unemb = net.embed/S
active_circuits = run.active_circuits
A_1 = run.A[1]

est_a_1 = torch.zeros(bs,z,d)

for i in range (d):
    est_a_1[:, :, i] = torch.einsum('btn,bn->bt', unemb[0, active_circuits], A_1[:,i*Dod:(i+1)*Dod])

est_a_1 = est_a_1[:,:,0] - est_a_1[:,:,1]
print((est_a_1).mean())



# %%
# Why isn't the on-indicator doing perfectly for z=1?

T = 500
D = 2*T
d = 4
Dod = D // d
S = 6

L = 5
z = 1

circ = RotSmallCircuits_4d(T, b=1)
net = CompInSup(D, L, S, circ)
run = net.run(L, z=z, bs=5, capped=True)
# %%


l = 2
print(run.A[l,0,:Dod] - run.A[l,0,Dod:2*Dod])



# %%
l = 2
print(run.pre_A[l,0,:Dod])
# %%
