#%% Setup
#   Setup

import numpy as np
import matplotlib.pyplot as plt
import torch


from sympy.ntheory import isprime
import torch

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

#%% Customizing the __repr__ method of torch.Tensor to save images
#   Customizing the __repr__ method of torch.Tensor to save images

try:
    original_repr 
except:
    original_repr = torch.Tensor.__repr__

def custom_repr(self):
    with torch.no_grad():
        if self.dim() == 2 and self.dtype == torch.float:
            image_data = self.cpu().numpy()
            filename = f"tensor_image_{list(self.shape)}.png"
            plt.imshow(image_data, cmap='gray', aspect='equal')
            plt.axis('off')
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.close()
        try:
            mean = self.mean().item()
        except:
            mean = "N/A"
        return f"shape={list(self.shape)}, mean={mean:.5} \n{original_repr(self)}"

torch.Tensor.__repr__ = custom_repr
#%% Resetting the __repr__ method
#   Resetting the __repr__ method
try:
    torch.Tensor.__repr__ = original_repr
except:
    pass

#%% Assignments for circuits in a superposition

def get_steps(S, Dod):

  primes_smaller_than_S = [num for num in range(2,S) if isprime(num)]

  step = 1
  while step * (S-1) < Dod:

    for p in primes_smaller_than_S:
      if step % p == 0:
        break
    else:
      yield step
      if not isprime(S):
        n=1
        while step * S**n * (S-1) < Dod:
          yield step * S**n
          n+=1
    step += 1


class MaxT(Exception):
  pass


def maxT(Dod = 500, S = 5):
  t = 0

  steps = get_steps(S, Dod)
  try:
    step = next(steps)
  except StopIteration:
    return int(t/S)

  shift=0
  i=0

  while True:

    if i + step*(S-1) >= Dod:
      shift += 1

      if shift >= step or shift + step*(S-1) >= Dod:
        try:
          step = next(steps)
        except StopIteration:
          return int(t)
        shift = 0

      i = shift

    for s in range(S):
      i += step
    t += 1


def comp_in_sup_assignment(T = 2000, Dod = 500, S = 5, device="cpu"):

  Dod = int(Dod)
  assignments = torch.zeros(T, Dod, dtype=torch.int64)
  compact_assignments = torch.zeros(T, S, dtype=torch.int64)

  if S == 1:
    i = 0
    for t in range(T):
      assignments[t,i] = 1
      compact_assignments[t,0] = i
      i += 1
      if i >= Dod:
        i = 0
    return assignments.to(device).float(), compact_assignments.to(device).int()

  steps = get_steps(S, Dod)
  step = next(steps)
  shift=0
  i=0

  for t in range(T):

    if i + step*(S-1) >= Dod:
      shift += 1

      if shift >= step or shift + step*(S-1) >= Dod:
        try:
          step = next(steps)
        except StopIteration:
          raise MaxT(f'Not enough step options. Max T = {t}')
        shift = 0

      i = shift

    for s in range(S):
      assignments[t,i] = 1
      compact_assignments[t,s] = i
      i += step

  return assignments.to(device).float(), compact_assignments.to(device).int()

def test_assignments(assignments, S):
  not_S =(assignments.sum(dim=1) != S).sum()
  if not_S == 0:
    print('Test 1 passed: Correct number of assigments for all circuits')
  else:
    print(f'Test 1 failed: Wrong number of assigments for {not_S} circuit(s)')

  overlap = (assignments.to(torch.float)) @ (assignments.to(torch.float).T) - S * torch.eye(T,device=device)
  if overlap.max() > 1:
    print('Test 2 failed: Overlap is above one, some pari of circuits')
  else:
    print('Test 2 passed: Overlap is max one, for all paris of circuits')


def slow_test_assignments(assignments, S):
  T = assignments.shape[0]

  passed_test_1 = True
  for t in range(T):
    if not assignments[t].sum() == S:
      print('Error: Wrong numbe of assigments for circuit ', t)
      passed_test = False

  if passed_test_1:
    print('Test passed: Correct number of assigments for all circuits')

  passed_test_2 = True
  for t in range(T):
    for u in range(t):
      if (assignments[t] * assignments[u]).sum() > 1:
        print('Error: Too high colition for circuits pair', u ,t)
        passed_test_2 = False

  if passed_test_2:
    print('Test passed: Overlap is max one, for all paris of circuits')

  if passed_test_1 and passed_test_2:
    return True
  else:
    return False
#%% Setting up the netork
#   Setting up the netork

class RunData:
   pass

class RotInSupNetwork:
    def __init__(self, Dod=1000, T=6000, S=5, device="cpu"):

        #Function parameters
        Dod = int(Dod) # Number of neurons in the large network devided by 4
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

        #Perserveing activationi indicators
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
        A[0, :, :Dod] = torch.einsum('bti->bi', assignments_1[active_circuits])
        A[0, :, Dod:2*Dod] = torch.einsum('bti->bi', assignments_2[active_circuits])
        A[0, :, 2*Dod:3*Dod] = torch.einsum('bt,bti->bi', (x[1,:,:,0] + 1, assignments_1[active_circuits]))
        A[0, :, 3*Dod:4*Dod] = torch.einsum('bt,bti->bi', (x[1,:,:,1] + 1, assignments_1[active_circuits])) 


        #Running the large netowrk: Layer 1
        A[1, :, :2*Dod] = - torch.relu(- A[0, :, :2*Dod] + 1) + 1 #implements min[1,x] = -ReLU(-x+1)+1
        A[1, :, 2*Dod:] = torch.relu(A[0, :, 2*Dod:])             #just copies over the values

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

#%% Small test
#   Small test
Dod=5
S=2
T=2
L=4
z=1
bs = 2

smal_test_net = RotInSupNetwork(Dod,T,S)
test_run = smal_test_net.run(L,z,bs)

for k in range(z):
    for b in range(bs):
        print(f'\n circuit {k} in batch {b}\n')
        for l in range(L):
            print(f'l={l}')
            print('x:    ', test_run.x[l,b,k])
            print('est_x:', test_run.est_x[l,b,k], '\n')


#%% Large test
#   Large test
Dod=500
S=5
T=3000
L=5
z=2
bs = 2

smal_test_net = RotInSupNetwork(Dod,T,S)
test_run = smal_test_net.run(L,z,bs)

for k in range(z):
    for b in range(bs):
        print(f'\n circuit {k} in batch {b}\n')
        for l in range(L):
            print(f'l={l}')
            print('x:    ', test_run.x[l,b,k])
            print('est_x:', test_run.est_x[l,b,k], '\n')


# %%
