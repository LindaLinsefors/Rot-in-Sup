#%% Reload classes_and_functions
#   Reload classes_and_functions

import numpy as np
import matplotlib.pyplot as plt
import torch

#Make sure networks.py and assignments.py are reloaded
import importlib, networks, assignments, classes_and_functions
importlib.reload(networks)
importlib.reload(assignments)
importlib.reload(classes_and_functions)

from assignments import (maxT, MaxT,
                         comp_in_sup_assignment,
                         expected_overlap_error, 
                         expected_squared_overlap_error, 
                         probability_of_overlap,
                         frequency_of_overlap)





class RunData:
   pass

class RotSmallCircuits:
    def __init__(self, T, b):
        self.T = T # Number of small circuits
        self.d = 3 # Number of neurons per small circuit
        
        #Small circuit rotations
        theta = torch.rand(T) * 2 * np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)

        self.mean_w = torch.zeros(3, 3)
        self.diff_w = torch.zeros(T, 3, 3)

        self.mean_w[0,0] = 1 + b
        self.mean_w[1,0] = 1 + b
        self.mean_w[2,0] = 1 + b

        self.diff_w[:,1,0] = - cos + sin
        self.diff_w[:,2,0] = - cos - sin
        self.diff_w[:,1,1] = cos
        self.diff_w[:,1,2] = -sin
        self.diff_w[:,2,1] = sin
        self.diff_w[:,2,2] = cos

        self.w = self.mean_w[None,:,:] + self.diff_w
        self.r = self.w[:, 1:, 1:]

        self.b = - torch.ones(3) * b

        self.rot = True

    def run(self, L, z, bs, active_circuits=None, initial_angle=None):
        """Run all small circuits on input random inputs"""

        a = torch.zeros(L+1, bs, z, 3)

        #Active circuits
        if active_circuits is None:  # Generating random circuits
            active_circuits = torch.randint(self.T, (bs, z))

            # Replace any duplicates with non-duplicates
            same = torch.zeros(bs, dtype=torch.bool)
            for i in range(z):
                for j in range(i):
                    same += (active_circuits[:,i] == active_circuits[:,j])
            n = same.sum()
            active_circuits[same] = torch.tensor(range(z*n), dtype=torch.int64).reshape(n, z) % self.T

        #Initial values
        if initial_angle is None:
            initial_angle = torch.rand(bs, z) * 2 * np.pi

        a[0, :, :, 0] = 1
        a[0, :, :, 1] = torch.cos(initial_angle) + 1
        a[0, :, :, 2] = torch.sin(initial_angle) + 1

        #Running the small circuits
        for l in range(L):
            a[l+1] = torch.relu(
                torch.einsum('btij,btj->bti', self.w[active_circuits], a[l]) 
                + self.b)

        return a, active_circuits
    



def expected_mse(T, Dod, l, b, z):
    if l == 0:
        return (0,0)
    
    mse_on = l * (z-1)/Dod + (l-1)*(1+b) * z*T/Dod**2
    mse_x =  l * (z-1)/Dod + (l-1)*(1+b) * z*T/Dod**2

    return (mse_on, mse_x)
# %%
