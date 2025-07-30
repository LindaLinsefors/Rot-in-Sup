'''
Various version of RotInSupNetwork

RotInSupNetwork_4d uses two 'on' indicators for each small network 
bringing the neurons per circuit up to d=4.

RotInSupNetwork_3d uses one 'on' indicator for each small network
bringing the neurons per circuit up to d=3, at the cost of higer error.
'''


import torch
from assignments import comp_in_sup_assignment
import numpy as np

#Default variables
Dod = 1000
T = 6000
S = 5
device = 'cpu'


class RunData:
   pass

class RotInSupNetwork_4d:
    def __init__(self, Dod=Dod, T=T, S=S, device=device):

        #Function parameters
        Dod = int(Dod) # Number of neurons in the large network divided by 4
        T = int(T) # Number of small circuits in superposition
        S = int(S) # Number of large network neurons used by each small circuit neuron

        #Embedding assignments
        assignments_1, compact_assignments_1 = comp_in_sup_assignment(T, Dod, S, device)
        shuffle = torch.randperm(T, device=device)
        assignments_2 = assignments_1[shuffle]
        compact_assignments_2 = compact_assignments_1[shuffle]

        #Small circuit rotations
        theta = torch.rand(T,device=device) * 2 * np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        r = torch.zeros(T, 2, 2, device=device)
        r[:,0,0] = cos
        r[:,0,1] = -sin
        r[:,1,0] = sin
        r[:,1,1] = cos

        #One vector
        one = torch.ones(2, device=device)

        #Large network weight matrices
        W1 = torch.zeros(4*Dod, 4*Dod, device=device)
        W2 = torch.zeros(4*Dod, 4*Dod, device=device)

        #Preserving activation indicators
        W1[:2*Dod, :2*Dod] = torch.eye(2*Dod, device=device)
        W2[:2*Dod, :2*Dod] = torch.eye(2*Dod, device=device)

        #Adding 2 to active circuit neurons
        W1[2*Dod:3*Dod, :Dod] = torch.eye(Dod, device=device) * 2
        W1[3*Dod:4*Dod, :Dod] = torch.eye(Dod, device=device) * 2
        W2[2*Dod:3*Dod, Dod:2*Dod] = torch.eye(Dod, device=device) * 2
        W2[3*Dod:4*Dod, Dod:2*Dod] = torch.eye(Dod, device=device) * 2

        #Removing rotated one
        rotated_one = torch.einsum('tn,tm,tij,j->nmi', (assignments_1, assignments_2, r, one))/S
        W1[2*Dod:3*Dod, Dod:2*Dod] = - rotated_one[:,:,0]
        W1[3*Dod:4*Dod, Dod:2*Dod] = - rotated_one[:,:,1]
        W2[2*Dod:3*Dod, :Dod] = - rotated_one[:,:,0].t()
        W2[3*Dod:4*Dod, :Dod] = - rotated_one[:,:,1].t()

        #Rotation
        all_rotations= torch.einsum('tn,tm,tij->nmij', (assignments_1, assignments_2, r))/S
        W1[2*Dod:3*Dod, 2*Dod:3*Dod] = all_rotations[:,:,0,0]
        W1[2*Dod:3*Dod, 3*Dod:4*Dod] = all_rotations[:,:,0,1]
        W1[3*Dod:4*Dod, 2*Dod:3*Dod] = all_rotations[:,:,1,0]
        W1[3*Dod:4*Dod, 3*Dod:4*Dod] = all_rotations[:,:,1,1]
        W2[2*Dod:3*Dod, 2*Dod:3*Dod] = all_rotations[:,:,0,0].t()
        W2[2*Dod:3*Dod, 3*Dod:4*Dod] = all_rotations[:,:,0,1].t()
        W2[3*Dod:4*Dod, 2*Dod:3*Dod] = all_rotations[:,:,1,0].t()
        W2[3*Dod:4*Dod, 3*Dod:4*Dod] = all_rotations[:,:,1,1].t()

        #Saving data to network object
        self.device = device
        self.Dod = Dod
        self.T = T
        self.S = S
        self.assignments_1 = assignments_1
        self.assignments_2 = assignments_2
        self.compact_assignments_1 = compact_assignments_1
        self.compact_assignments_2 = compact_assignments_2
        self.r = r
        self.W1 = W1
        self.W2 = W2

        #Empty list for storing run data later
        self.runs = []
        self.run_by_name = {}


    def run(self, L = 2, z = 2, bs = 2, run_name = None):

        #Function parameters
        L = int(L) # Number of layers
        z = int(z) # Number of circuits in superposition
        bs = int(bs) # Batch size

        #Import network data as local variables
        device = self.device
        Dod = self.Dod
        T = self.T
        S = self.S
        assignments_1 = self.assignments_1
        assignments_2 = self.assignments_2
        r = self.r
        W1 = self.W1
        W2 = self.W2

        #Bias
        B = torch.zeros(4*Dod, device=device)
        B[2*Dod:] = -1

        #Neuron activations (batched)
        A = torch.zeros(L, bs, 4*Dod, device=device)
        x = torch.zeros(L, bs, z, 2, device=device)
        est_x = torch.zeros(L, bs, z, 2, device=device)

        #Inputs
        active_circuits = torch.randint(T, (bs, z), device=device)
        initial_angle = torch.rand(bs, z, device=device) * 2 * np.pi
        x[0, :, :, 0] = torch.cos(initial_angle)
        x[0, :, :, 1] = torch.sin(initial_angle)
        
        est_x[0] = x[0]
        
        #Running the small circuits
        for l in range(1,L):
            x[l] = torch.einsum('btij,btj->bti', r[active_circuits], x[l-1])

        #Large network initial values
        A[0, :,    :Dod  ] = torch.einsum('bti->bi', assignments_1[active_circuits])
        A[0, :, Dod:2*Dod] = torch.einsum('bti->bi', assignments_2[active_circuits])
        A[0, :, 2*Dod:3*Dod] = torch.einsum('bt,bti->bi', (x[1,:,:,0], assignments_1[active_circuits]))
        A[0, :, 3*Dod:     ] = torch.einsum('bt,bti->bi', (x[1,:,:,1], assignments_1[active_circuits])) 


        #Running the large network: Layer 1
        A[1, :, :2*Dod] = - torch.relu(- A[0, :, :2*Dod] + 1) + 1 #implements min[1,x] = -ReLU(-x+1)+1
        A[1, :, 2*Dod:3*Dod] = torch.relu(A[0, :, 2*Dod:3*Dod] + 1) - 1 + A[1, :, :Dod]
        A[1, :, 3*Dod:     ] = torch.relu(A[0, :, 3*Dod:     ] + 1) - 1 + A[1, :, :Dod]

        #Running the large network: All other layers
        for l in range(2,L):
            if l%2 == 1: # Odd layers
                A[l] = torch.relu(torch.einsum('ij,bj->bi', W1, A[l-1]) + B[None,:])
            else: # Even layers
                A[l] = torch.relu(torch.einsum('ij,bj->bi', W2, A[l-1]) + B[None,:])

        #Extracting estimates for x in each layer
        for l in range(1,L):
            if l%2 == 1: # Odd layers
                est_x[l,:,:,0] = torch.einsum('btn,bn->bt', 
                                            assignments_1[active_circuits], A[l, :, 2*Dod:3*Dod])/S - 1
                est_x[l,:,:,1] = torch.einsum('btn,bn->bt',  
                                            assignments_1[active_circuits], A[l, :, 3*Dod:4*Dod])/S - 1
            else: # Even layers
                est_x[l,:,:,0] = torch.einsum('btn,bn->bt', 
                                            assignments_2[active_circuits], A[l, :, 2*Dod:3*Dod])/S - 1
                est_x[l,:,:,1] = torch.einsum('btn,bn->bt', 
                                            assignments_2[active_circuits], A[l, :, 3*Dod:4*Dod])/S - 1

        # Saving run data
        run = RunData()
        run.L = L
        run.z = z
        run.bs = bs
        run.active_circuits = active_circuits
        run.x = x
        run.est_x = est_x
        run.A = A
        run.name = run_name

        self.runs.append(run)

        if run_name is not None:
            if run_name in self.run_by_name:
                print(f"Warning: Overwriting existing run with name '{run_name}'")
            self.run_by_name[run_name] = run

        return run
    


