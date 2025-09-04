#%% Reload classes_and_functions
#   Reload classes_and_functions

import numpy as np
import matplotlib.pyplot as plt
import torch

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





class RunData:
   '''Empty class to store the data from a run of the network.'''
   pass




def random_active_circuits(T, bs, z):
    '''Generates z non identical integers from 0 to T-1 for each of the bs batches.''' 

    active_circuits = torch.randint(T, (bs, z))

    # Replace any duplicates with non-duplicates
    same = torch.zeros(bs, dtype=torch.bool)
    for i in range(z):
        for j in range(i):
            same += (active_circuits[:,i] == active_circuits[:,j])
    n = same.sum()
    active_circuits[same] = torch.tensor(range(z*n), dtype=torch.int64).reshape(n, z) % T

    return active_circuits


def random_rotation_matrices(T):
    """
    Generates T random 2D rotation matrices.
    """

    theta = torch.rand(T) * 2 * np.pi
    cos = torch.cos(theta)
    sin = torch.sin(theta)

    r = torch.zeros(T, 2, 2)
    r[:,0,0] = cos;  r[:,0,1] = -sin
    r[:,1,0] = sin;  r[:,1,1] = cos

    return r



class RotSmallCircuits_3D:
    """
    Small circuits that rotate a 2D vector and have one on-indicator neuron.

    If there is any noise in the input to the indicator neuron, this noise is amplified,
    causing these circuits to be unstable. Oops!

    Neuron 0 is the on-indicator neuron.
    Neuron 1 and 2 are the 2D vector that is being rotated.
    """

    def __init__(self, T, b):
        self.T = T # Number of small circuits
        self.d = 3 # Number of neurons per small circuit
        
        # b is the magnitude of the bias (on all three neurons).
        self.b = - torch.ones(3) * b # Biases for the three neurons

        self.mean_w = torch.zeros(3, 3) #Part of the weights that is the same for all circuits
        self.diff_w = torch.zeros(T, 3, 3) #Part of the weights that is different for each circuit

        #Overcomes the bias and duplicates the value of the on-indicator to the next layer.
        #Unfortunatly unstabel to noise!
        self.mean_w[0,0] = 1 + b

        self.mean_w[-2: ,0] = 1 + b #Shift to overcome the bias and pass though the ReLUs

        self.r = random_rotation_matrices(T) # The rotation part of the circuits
        self.is_rot = True # indicate that this circuit does rotations

        #Compensates for vectors being centered around (1,1) instead of (0,0)
        self.diff_w[:, -2:, 0] = - self.r[:,:,0] - self.r[:,:,1]

        #Rotation
        self.diff_w[:, -2:, -2:] = self.r

        #The combined weights for each circuit.
        self.w = self.mean_w[None,:,:] + self.diff_w

    
        


    def run(self, L, z, bs, active_circuits=None, initial_angle=None):
        """
        Run all active_circuits on input random inputs and return 
        the activations of all layers.
        """

        a = torch.zeros(L+1, bs, z, 3)

        #Active circuits
        if active_circuits is None:  # Generating random circuits
            active_circuits = random_active_circuits(self.T, bs, z)

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
    




class RotSmallCircuits_4D:
    """
    Small circuits that uses the first two neurons to implement a step function, 
    to track if the circuit is active or not, and the last two neurons to rotate a 2D vector.
    """

    def __init__(self, T, b):
        self.T = T # Number of small circuits
        self.d = 4 # Number of neurons per small circuit

        # b is the magnitude of the bias on the rotation neurons.
        self.b = torch.tensor([-0.5, -1.5, -b, -b]) # Biases for the four neurons

        self.mean_w = torch.zeros(4, 4) #Part of the weights that is the same for all circuits
        self.diff_w = torch.zeros(T, 4, 4) #Part of the weights that is different for each circuit
        
        #These toghether with the bias for the two first neruons implements a step function.
        self.mean_w[0:1, 0] = 2
        self.mean_w[0:1, 1] = -2
        #(a0 - a1) = ReLU(2*(a0 - a1) - 0.5) - ReLU(2*(a0 - a1) - 1.5) 

        #Shift to overcome the bias and pass though the ReLUs
        self.mean_w[-2:, 0] = (1 + b) 
        self.mean_w[-2:, 1] = -(1 + b)

        self.r = random_rotation_matrices(T) # The rotation part of the circuits
        self.is_rot = True # indicate that this circuit does rotations

        #Compensates for vectors being centered around (1,1) instead of (0,0)
        self.diff_w[:, -2:, 0] = - self.r[:,:,0] - self.r[:,:,1]
        self.diff_w[:, -2:, 1] = + self.r[:,:,0] + self.r[:,:,1]

        #Rotation
        self.diff_w[:, -2:, -2:] = self.r

        #The combined weights for each circuit.
        self.w = self.mean_w[None,:,:] + self.diff_w




    def run(self, L, z, bs, active_circuits=None, initial_angle=None):
        """Run all small circuits on input random inputs"""

        a = torch.zeros(L+1, bs, z, 3)

        #Active circuits
        if active_circuits is None:  # Generating random circuits
            active_circuits = random_active_circuits(self.T, bs, z)

        #Initial values
        if initial_angle is None:
            initial_angle = torch.rand(bs, z) * 2 * np.pi

        a[0, :, :, 0] = 1.5
        a[0, :, :, 1] = 0.5
        a[0, :, :, 2] = torch.cos(initial_angle) + 1
        a[0, :, :, 3] = torch.sin(initial_angle) + 1

        #Running the small circuits
        for l in range(L):
            a[l+1] = torch.relu(
                torch.einsum('btij,btj->bti', self.w[active_circuits], a[l]) 
                + self.b)

        return a, active_circuits









class CompInSup:
    def __init__(self, D, L, S, small_circuits, u_correction=None):

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
        self.u_correction = u_correction # Negative correction for unembedding

        # Separate the small circuit weights into average (mean_w) 
        # and distance from average (diff_w).
        try:
            mean_w = small_circuits.mean_w
            diff_w = small_circuits.diff_w
        except AttributeError:
            mean_w = w.mean(dim=0)
            diff_w = w - mean_w[None,:,:]

        # Is the small circuits doing rotation?
        try:
            self.is_rot = small_circuits.is_rot
        except AttributeError:
            self.is_rot = False

        embed = torch.zeros(L, T, Dod) # Embedding vectors
        unemb = torch.zeros(L, T, Dod)  # Unembedding vectors
        assign = torch.zeros(L, T, S, dtype=torch.int64) # Neuron assignments

        # Create the embeddings and assignments for each layer. 
        # embed and assign contain the same information, but in different formats.
        embed[0], assign[0] = comp_in_sup_assignment(T, Dod, S)
        for l in range(1, L):
            shuffle = torch.randperm(T)
            embed[l] = embed[0][shuffle]
            assign[l] = assign[0][shuffle]
        
        # Apply standard correction if none is given.
        if u_correction is None: 
            p = probability_of_overlap(T, Dod, S)
            u_correction = p/((S-p)*S)

        # Create the unembedding vectors. 
        # If standard correction is used, this should cause the expected dotproduct between 
        # embed and unembed for diffrent circuits is zero.
        unemb = - torch.ones(L, T, Dod) * u_correction
        unemb += embed * (1/S + u_correction)

        W = torch.zeros(L, D, D) # Weight matrices for the large network
        W[0] = torch.eye(D) # First layer is always the identity


        '''There are three sets of weight matrices, to compare different implementations.'''

        # Used for: not "capped" and not "split":
        # The corrected unemb is used for the entire weight matrix.
        W1 = torch.zeros(L, D, D)
        W1[0] = torch.eye(D)

        for l in range(1,L):
            temp_W = torch.einsum('tn,tij,tm->ijnm', unemb[l], w, embed[l-1])

            for i in range(d):
                for j in range(d):
                    W1[l, i*Dod:(i+1)*Dod, j*Dod:(j+1)*Dod] = temp_W[i,j]


        # Used for: not "capped" and "split":
        # The corrected unemb is used only to embed mean_w.
        W2 = torch.zeros(L, D, D)
        W2[0] = torch.eye(D)

        for l in range(1,L):
            temp_W = (torch.einsum('tn,ij,tm->ijnm', unemb[l], mean_w, embed[l-1])
                      +torch.einsum('tn,tij,tm->ijnm', embed[l]/S, diff_w, embed[l-1]))
            
            for i in range(d):
                for j in range(d):
                    W2[l, i*Dod:(i+1)*Dod, j*Dod:(j+1)*Dod] = temp_W[i,j]

        
        # Used for: "capped"
        # The corrected unemb is not used at all.
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

        d = self.small_circuits.d # Number of neurons in each small circuit
        B = self.B # Biases for each layer
        Dod = self.Dod # D/d, where D is the number of neurons in the large network

        if not capped and not split:
            W = self.W1
            unemb = self.unemb
        elif not capped and split:
            W = self.W2
            unemb = self.unemb
        else:
            W = self.W3
            unemb = self.embed/self.S

        if initial_angle is not None:
            a, active_circuits = self.small_circuits.run(L, z, bs, active_circuits, initial_angle)
        else:
            a, active_circuits = self.small_circuits.run(L, z, bs, active_circuits)

        A = torch.zeros(L+1, bs, self.D)
        pre_A = torch.zeros(L+1, bs, self.D)

        temp_A = torch.einsum('btn,bti->ibn', self.embed[0,active_circuits],a[1])
        for i in range(d):
            A[0,:,i*Dod:(i+1)*Dod] = temp_A[i]
        pre_A[0] = A[0]

        for l in range(L):
            pre_A[l+1] = torch.einsum('nm,bm->bn', W[l], A[l]) + B[l]
            A[l+1] = torch.relu(pre_A[l+1])
            #A[l+1] = pre_A[l+1]

        est_a = torch.zeros(L+1, bs, z, d)
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
    


def expected_mse_rot(T, Dod, l, b, z):
    '''
    Theoretically predicted Mean Squared Error for the rotation part of the network,
    assuming the indicator part or the network works perfectly.
    '''

    if l == 0:
        return (0,0)
    
    mse =  l * (z-1)/Dod + (l-1)*(1+b) * z*T/Dod**2
    return mse



def plot_mse_rot(L, labels, runs, title, expected=None, y_max=None):
    """Plot the mean squared error for a set of runs."""
    
    mse_x = []
    mse_on = []
    for run in runs:
        mse_x.append((run.x - run.est_x).pow(2).mean(dim=(1, 2)).sum(dim=-1).cpu().numpy())
        mse_on.append((run.a[:,:,:,0] - run.est_a[:,:,:,0]).pow(2).mean(dim=(1, 2)).cpu().numpy())

  
    fig = plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    for i, label in enumerate(labels):
        line, = plt.plot(mse_on[i], marker='o', label=label)
        if expected is not None:
            plt.plot([expected[i][l] for l in range(L+1)], 
                     linestyle='--', color=line.get_color(), marker='x')
    plt.title('Active Circuits On-Indicator')
    plt.xlabel('Layer')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    if y_max is not None:
        plt.ylim(0, y_max) 
    plt.legend()

    plt.subplot(1, 2, 2)
    for i, label in enumerate(labels):
        line, = plt.plot(mse_x[i], marker='o', label=label)
        if expected is not None:
            plt.plot([expected[i][l] for l in range(L+1)], 
                     linestyle='--', color=line.get_color(), marker='x')
    plt.title('Active Circuits Rotated Vector')
    plt.xlabel('Layer')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.legend()

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Make room for the title

    plt.show()


def get_inactive_circuits(active_circuits, T):
    """
    Given a tensor listing the active circuits for each batch, 
    returns a tensor listing the inactive circuits for each batch.
    """

    bs, z = active_circuits.shape # bs is batch size, z is number of active circuits

    all_idx = torch.arange(T).expand(bs, T)          # [bs, T]
    mask = torch.ones(bs, T, dtype=torch.bool)       # [bs, T]
    mask[torch.arange(bs).unsqueeze(1), active_circuits] = False

    inactive_circuits = all_idx[mask].view(bs, T - z)  #[bs, T-z]

    return inactive_circuits




# %%
