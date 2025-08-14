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


class CompInSup:
    def __init__(self, D, L, S, small_circuits, correction=None):

        d = int(small_circuits.d)  # Number of neurons in each small circuit
        T = small_circuits.T # Number of small circuits
        D = int(D) # Number of neurons in the large network
        Dod = D // d # D/d
        w = small_circuits.w # Weights of the small circuits

        self.T = T # Number of small circuits
        self.D = D # Number of neurons in the large network
        self.Dod = Dod # D/d
        self.L = L # Number of layers in the large network
        self.S = S # Number of large network neurons used by each small circuit neuron
        self.small_circuits = small_circuits
        self.correction = correction # Negative correction for unembedding

        try:
            mean_w = small_circuits.mean_w
            diff_w = small_circuits.diff_w
        except AttributeError:
            mean_w = w.mean(dim=0)
            diff_w = w - mean_w[None,:,:]

        try:
            self.rot = small_circuits.rot
        except AttributeError:
            self.rot = False


        embed = torch.zeros(L, T, Dod)
        unemb = torch.zeros(L, T, Dod)
        assign = torch.zeros(L, T, S, dtype=torch.int64)

        embed[0], assign[0] = comp_in_sup_assignment(T, Dod, S)
        for l in range(1, L):
            shuffle = torch.randperm(T)
            embed[l] = embed[0][shuffle]
            assign[l] = assign[0][shuffle]
        
        if correction is None:
            p = probability_of_overlap(T, Dod, S)
            correction = p/((S-p)*S)
        unemb = - torch.ones(L, T, Dod) * correction
        unemb += embed * (1/S + correction)

        W = torch.zeros(L, D, D)
        W[0] = torch.eye(D)

        # Used for: not capped and not split:
        W1 = torch.zeros(L, D, D)
        W1[0] = torch.eye(D)

        for l in range(1,L):
            temp_W = torch.einsum('tn,tij,tm->ijnm', unemb[l], w, embed[l-1])

            for i in range(d):
                for j in range(d):
                    W1[l, i*Dod:(i+1)*Dod, j*Dod:(j+1)*Dod] = temp_W[i,j]


        # Used for: not capped and split:
        W2 = torch.zeros(L, D, D)
        W2[0] = torch.eye(D)

        for l in range(1,L):
            temp_W = (torch.einsum('tn,ij,tm->ijnm', unemb[l], mean_w, embed[l-1])
                      +torch.einsum('tn,tij,tm->ijnm', embed[l]/S, diff_w, embed[l-1]))
            
            for i in range(d):
                for j in range(d):
                    W2[l, i*Dod:(i+1)*Dod, j*Dod:(j+1)*Dod] = temp_W[i,j]

        
        # Used for: capped
        W3 = torch.zeros(L, D, D)
        W3[0] = torch.eye(D)

        for l in range(1,L):

            capped_embed = torch.einsum('tn,tm->nm',embed[l],embed[l-1])
            capped_embed.clamp_(max=1.0)


            above_diag = torch.triu(torch.ones(T, T), diagonal=1).bool()
            every_unwanted_interaction = torch.einsum('tn,nm,um->tu', embed[l], capped_embed, embed[l-1])[above_diag].sum()
            every_possible_interaction = T*(T-1)/2 * S*S
            capped_corr_1 = every_unwanted_interaction/(every_possible_interaction-every_unwanted_interaction)

            ces = capped_embed.sum()
            capped_corr_2 = ces/(Dod**2-ces) #Alternative correction value.

            capped_embed -= (torch.ones_like(capped_embed) - capped_embed) * capped_corr_1

            temp_W = (torch.einsum('tn,tij,tm->ijnm', embed[l], diff_w, embed[l-1])
                      + torch.einsum('nm,ij->ijnm', capped_embed, mean_w)
                     )/S
            
            for i in range(d):
                for j in range(d):
                    W3[l, i*Dod:(i+1)*Dod, j*Dod:(j+1)*Dod] = temp_W[i,j]


        #First bias is zero, the rest are the biases of the small circuits
        B = torch.zeros(L, D)
        for i in range(d):
            B[1:, i*Dod:(i+1)*Dod] = small_circuits.b[i]


        self.B = B
        self.embed = embed
        self.unemb = unemb
        self.assign = assign

        self.W1 = W1
        self.W2 = W2
        self.W3 = W3

    def run(self, L, z, bs, active_circuits=None, initial_angle=None, 
            capped=False, split=False):

        d = self.small_circuits.d
        B = self.B
        Dod = self.Dod

        if not capped and not split:
            W = self.W1
            unemb = self.unemb
        elif not capped and split:
            W = self.W2
            unemb = self.unemb
        else:
            W = self.W3
            unemb = self.embed/self.S

        a, active_circuits = self.small_circuits.run(L, z, bs, active_circuits, initial_angle)

        A = torch.zeros(L+1, bs, self.D)
        pre_A = torch.zeros(L+1, bs, self.D)

        [A[0,:,:Dod], A[0,:,Dod:2*Dod], A[0,:,2*Dod:]] = torch.einsum('btn,bti->ibn', self.embed[0,active_circuits],a[1])
        pre_A[0] = A[0]

        for l in range(L):
            pre_A[l+1] = torch.einsum('nm,bm->bn', W[l], A[l]) + B[l]
            A[l+1] = torch.relu(pre_A[l+1])
            #A[l+1] = pre_A[l+1]

        est_a = torch.zeros(L+1, bs, z, 3)
        est_a[0] = a[0]
        for l in range(L):
            for i in range(d):
                est_a[l+1, :, :, i] = torch.einsum('btn,bn->bt', unemb[l, active_circuits], A[l+1,:,i*Dod:(i+1)*Dod])

        est_x = est_a[:, :, :, 1:] - est_a[:, :, :, 0][:, :, :, None]

        run = RunData()
        run.a = a
        run.est_a = est_a
        run.active_circuits = active_circuits
        run.A = A
        run.pre_A = pre_A

        if self.rot:
            x = a[:,:,:,1:] - 1
            est_x = est_a[:, :, :, 1:] - est_a[:, :, :, 0][:, :, :, None]
            on = a[:,:,:,0]
            est_on = est_a[:, :, :, 0]

            run.x = x
            run.est_x = est_x
            run.on = on
            run.est_on = est_on

        return run