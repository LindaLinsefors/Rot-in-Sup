'''
Various version of Rotation in Superpossition.

These implentations uses indicator neurons with a fixed embedding so they 
don't have to be re-calculated from layer to layer. This means less noise,
but also a less general implementation.

RotInSupNetwork_4d uses two 'on' indicators for each small network 
bringing the neurons per circuit up to d=4.

RotInSupNetwork_3d uses one 'on' indicator for each small network
bringing the neurons per circuit up to d=3, at the cost of higer error.
'''


import torch
from assignments import comp_in_sup_assignment, expected_overlap_error, expected_squared_overlap_error
import numpy as np

#Default variables
Dod = 1000
T = 6000
S = 5
device = 'cpu'

class SmallCircuits:
    def __init__(self, T=T, device=device):
        self.T = T
        self.device = device
        
        #Small circuit rotations
        theta = torch.rand(T,device=device) * 2 * np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        self.r = torch.zeros(T, 2, 2, device=device)
        self.r[:,0,0] = cos
        self.r[:,0,1] = -sin
        self.r[:,1,0] = sin
        self.r[:,1,1] = cos

    def run(self, L, z, bs, active_circuits=None):
        """Run all small circuits on input random inputs"""

        device = self.device
        x = torch.zeros(L, bs, z, 2, device=device)
        T = self.T
        r = self.r

        #Inputs
        if active_circuits is None:  # Generating random circuits
            active_circuits = torch.randint(T, (bs, z), device=device)

            # Replace any duplicates with non-duplicates
            same = torch.zeros(bs, dtype=torch.bool, device=device)
            for a in range(z):
                for b in range(a):
                    same += (active_circuits[:,a] == active_circuits[:,b])
            n = same.sum()
            active_circuits[same] = torch.tensor(range(z*n), dtype=torch.int64, device=device).reshape(n, z) % T

        initial_angle = torch.rand(bs, z, device=device) * 2 * np.pi
        x[0, :, :, 0] = torch.cos(initial_angle)
        x[0, :, :, 1] = torch.sin(initial_angle)
        
        #Running the small circuits
        for l in range(1,L):
            x[l] = torch.einsum('btij,btj->bti', r[active_circuits], x[l-1])

        return x, active_circuits


class RunData:
   pass

class RotInSupNetwork_4d:
    def __init__(self, Dod=Dod, T=T, S=S, device=device):

        #Function parameters
        Dod = int(Dod) # Number of neurons in the large network divided by 4
        T = int(T) # Number of small circuits in superposition
        S = int(S) # Number of large network neurons used by each small circuit neuron

        #Small circuits
        small_circuits = SmallCircuits(T, device)
        r = small_circuits.r

        #Embedding assignments
        assignments_1, compact_assignments_1 = comp_in_sup_assignment(T, Dod, S, device)
        shuffle = torch.randperm(T, device=device)
        assignments_2 = assignments_1[shuffle]
        compact_assignments_2 = compact_assignments_1[shuffle]

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
        self.small_circuits = small_circuits
        self.d = 4

        #Empty list for storing run data later
        self.runs = []
        self.run_by_name = {}


    def run(self, L = 2, z = 2, bs = 2, run_name = None, active_circuits = None, ideal=False):

        #Function parameters
        L = int(L) # Number of layers
        if active_circuits is None:
            z = int(z) # Number of circuits in superposition
            bs = int(bs) # Batch size
        else:
            bs, z = active_circuits.shape

        #Import network data as local variables
        device = self.device
        Dod = self.Dod
        S = self.S
        assignments_1 = self.assignments_1
        assignments_2 = self.assignments_2
        W1 = self.W1
        W2 = self.W2
        small_circuits = self.small_circuits

        #Run small circuits
        x, active_circuits = small_circuits.run(L, z, bs, active_circuits)

        #Bias
        B = torch.zeros(4*Dod, device=device)
        B[2*Dod:] = -1 

        #Large network initial values
        A = torch.zeros(L, bs, 4*Dod, device=device)
        A[0, :,    :Dod  ] = torch.einsum('bti->bi', assignments_1[active_circuits])
        A[0, :, Dod:2*Dod] = torch.einsum('bti->bi', assignments_2[active_circuits])
        A[0, :, 2*Dod:3*Dod] = torch.einsum('bt,bti->bi', (x[1,:,:,0], assignments_1[active_circuits]))
        A[0, :, 3*Dod:     ] = torch.einsum('bt,bti->bi', (x[1,:,:,1], assignments_1[active_circuits])) 


        #Running the large network: Layer 1
        A[1, :, :2*Dod] = - torch.relu(- A[0, :, :2*Dod] + 1) + 1 #implements min[1,x] = -ReLU(-x+1)+1
        A[1, :, 2*Dod:3*Dod] = torch.relu(A[0, :, 2*Dod:3*Dod] + 1) - 1 + A[1, :, :Dod]
        A[1, :, 3*Dod:     ] = torch.relu(A[0, :, 3*Dod:     ] + 1) - 1 + A[1, :, :Dod]

        

        #Running the large network: All other layers
        if not ideal:
            for l in range(2,L):
                if l%2 == 1: # Odd layers
                    A[l] = torch.relu(torch.einsum('ij,bj->bi', W1, A[l-1]) + B[None,:])
                else: # Even layers
                    A[l] = torch.relu(torch.einsum('ij,bj->bi', W2, A[l-1]) + B[None,:])


        if ideal:
            mask_1 = torch.einsum('bti->bi', assignments_1[active_circuits]).clamp(max=1.0)
            mask_2 = torch.einsum('bti->bi', assignments_2[active_circuits]).clamp(max=1.0)
            for l in range(2,L):
                if l%2 == 1: # Odd layers
                    A[l] = torch.einsum('ij,bj->bi', W1, A[l-1]) + B[None,:]
                    A[l,:,2*Dod:3*Dod] *= mask_1
                    A[l,:,3*Dod:] *= mask_1
                else: # Even layers
                    A[l] = torch.einsum('ij,bj->bi', W2, A[l-1]) + B[None,:]   
                    A[l,:,2*Dod:3*Dod] *= mask_2
                    A[l,:,3*Dod:] *= mask_2

        #Extracting estimates for x in each layer
        est_x = torch.zeros(L, bs, z, 2, device=device)
        est_x[0] = x[0]

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
        run.net = self

        self.runs.append(run)

        if run_name is not None:
            if run_name in self.run_by_name:
                print(f"Warning: Overwriting existing run with name '{run_name}'")
            self.run_by_name[run_name] = run

        return run
    


class RotInSupNetwork_3d:
    def __init__(self, Dod=Dod, T=T, S=S, L=2, balance=True, improved_balance=True, device=device):

        #Function parameters
        Dod = int(Dod) # Number of neurons in the large network divided by 4
        T = int(T) # Number of small circuits in superposition
        S = int(S) # Number of large network neurons used by each small circuit neuron
        L = int(L) # Number of layers before W repeats

        #Small circuits
        small_circuits = SmallCircuits(T, device)
        r = small_circuits.r

        #Embedding assignments for the on indicator
        assignments_on = torch.randn(T, Dod, device=device)
        assignments_on = assignments_on / assignments_on.norm(dim=1, keepdim=True)

        #Embedding assignments for the vector values
        assignments = torch.zeros(L, T, Dod, device=device)
        compact_assignments = torch.zeros(L, T, S, device=device, dtype=torch.int32)

        assignments[0], compact_assignments[0] = comp_in_sup_assignment(T, Dod, S, device)

        for l in range(1,L):
            shuffle = torch.randperm(T, device=device)
            assignments[l] = assignments[0,shuffle]
            compact_assignments[l] = compact_assignments[0,shuffle]

        #Used for corelated computations only
        if balance:
            if not improved_balance:
                #Slightly negative for non assigned neurons s.t. balanced_assignments.mean()=0
                balanced_assignments = assignments * (1 + S/(Dod - S)) - torch.ones_like(assignments) * S/(Dod - S) 
            else:
                #Taking into account that the expected overlap error slighly less than S/Dod
                e = expected_overlap_error(T, Dod, S)
                x = e / (1-e)
                balanced_assignments = assignments * (1 + x) - torch.ones_like(assignments) * x
        else:
            balanced_assignments = assignments

        #Used as index to get assigment from previous layer
        previous = torch.roll(torch.arange(0, L, device=device), shifts=1)

        #One vector
        one = torch.ones(2, device=device)

        #Large network weight matrices
        W = torch.zeros(L, 3*Dod, 3*Dod, device=device)

        #Preserving activation indicators
        W[:, :Dod, :Dod] = torch.eye(Dod, device=device)[None, :, :]

        #Adding 2 to active circuit neurons and suptracting rotation of one-vector
        (W[:, Dod:2*Dod, :Dod], 
         W[:, 2*Dod:,    :Dod]) = (
                    2 * torch.einsum('ltn,tm,i->ilnm', (balanced_assignments, assignments_on, one)) 
                    - torch.einsum('ltn,tm,tij,j->ilnm', (assignments, assignments_on, r, one)) )

        #Rotating
        ((W[:, Dod:2*Dod, Dod:2*Dod], W[:, Dod:2*Dod, 2*Dod:]), 
        ( W[:, 2*Dod:,    Dod:2*Dod], W[:, 2*Dod:,    2*Dod:])) = (
                    torch.einsum('ltn,ltm,tij->ijlnm', (assignments, assignments[previous], r)) / S)

        #Saving data to network object
        self.device = device
        self.Dod = Dod
        self.T = T
        self.S = S
        self.assignments_on = assignments_on
        self.assignments = assignments
        self.balanced_assignments = balanced_assignments
        self.compact_assignments = compact_assignments
        self.r = r
        self.W = W
        self.small_circuits = small_circuits
        self.L_W = L # Number of layers before W repeatss
        self.d = 3

        #Empty list for storing run data later
        self.runs = []
        self.run_by_name = {}

    def run(self, L = 2, z = 2, bs = 2, run_name = None, active_circuits=None):

        #Function parameters
        L = int(L) # Number of layers
        if active_circuits is None:
            z = int(z) # Number of circuits in superposition
            bs = int(bs) # Batch size
        else:
            bs, z = active_circuits.shape

        #Import network data as local variables
        device = self.device
        Dod = self.Dod
        S = self.S
        assignments_on = self.assignments_on
        assignments = self.assignments
        balanced_assignments = self.balanced_assignments
        W = self.W
        small_circuits = self.small_circuits
        L_W = self.L_W
        
        #Run small circuits
        x, active_circuits = small_circuits.run(L, z, bs)

        #Large network initial values
        A = torch.zeros(L, bs, 3*Dod, device=device)
        A[0, :, :Dod] = torch.einsum('bti->bi', assignments_on[active_circuits])
        A[0, :, Dod:2*Dod], A[0, :, Dod*2:] = torch.einsum('btn,bti->ibn', (assignments[1%L, active_circuits], x[1]))

        est_x = torch.zeros(L, bs, z, 2, device=device)
        est_x[0] = x[0]

        #Running the large network: Layer 1
        A[1] = torch.relu(A[0] + 1) - 1 #Coppying over everything from previous layer

        est_x[1,:,:,0] = torch.einsum('btn,bn->bt', (assignments[1%L, active_circuits], A[1, :, Dod:2*Dod])) / S
        est_x[1,:,:,1] = torch.einsum('btn,bn->bt', (assignments[1%L, active_circuits], A[1, :, Dod*2:])) / S

        #Running the large network: Layer 2
        A[2, :, :Dod] = A[1, :, :Dod] #Coppying over activation values from previous layer
        A[2, :, Dod:] = torch.relu(torch.einsum('nm,bm->bn', (W[2%L_W, Dod:, Dod:], A[1, :, Dod:]))  #Rotating
                        + 2 * torch.tile(torch.einsum('tn,tm,bm->bn', (balanced_assignments[2%L_W], assignments_on, A[1, :, :Dod])),
                                        dims = (2,)) #Add 2 to active
                        - 1) #bias = -1
                                    
        #All other layers
        for l in range(3,L):
            A[l] = torch.einsum('nm,bm->bn', (W[l%L_W], A[l-1]))
            A[l, :, Dod:] = torch.relu(A[l, :, Dod:] - 1)


        for l in range(2,L):
            est_x[l,:,:,0] = torch.einsum('btn,bn->bt', (assignments[l%L_W, active_circuits], A[l, :, Dod:2*Dod])) / S - 1
            est_x[l,:,:,1] = torch.einsum('btn,bn->bt', (assignments[l%L_W, active_circuits], A[l, :, Dod*2:])) / S - 1
            

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
        run.net = self

        self.runs.append(run)

        if run_name is not None:
            if run_name in self.run_by_name:
                print(f"Warning: Overwriting existing run with name '{run_name}'")
            self.run_by_name[run_name] = run

        return run
    



def semi_correlated(L):
    return (L//2)**2 + ((L+1)//2)**2

def expected_mse_4d(T, Dod, L, z, naive=True):
    if L == 0:
        return 0
    else:
        if naive == False:
            esor = expected_squared_overlap_error(T, Dod, S, naive=False)
        else :
            esor = 1/Dod

        return T * esor**2 * ((L-1)**2 + (z-1)*semi_correlated(L-1)) + esor * (z-1)*semi_correlated(L)

def expected_mse_3d(T, Dod, L, z, naive=True):
    if L == 0:
        return 0
    else:
        if naive == False:
            esor = expected_squared_overlap_error(T, Dod, S, naive=False)
        else :
            esor = 1/Dod


        return T * esor**2 * (8+1+1+2) * z * (L-1) + esor * (z-1) * (L + (L-1) * (8+1+1+2))