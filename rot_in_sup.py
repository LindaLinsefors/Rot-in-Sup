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

def get_steps(S, D_over_d):

  primes_smaller_than_S = [num for num in range(2,S) if isprime(num)]

  step = 1
  while step * (S-1) < D_over_d:

    for p in primes_smaller_than_S:
      if step % p == 0:
        break
    else:
      yield step
      if not isprime(S):
        n=1
        while step * S**n * (S-1) < D_over_d:
          yield step * S**n
          n+=1
    step += 1


class MaxT(Exception):
  pass


def maxT(D_over_d = 500, S = 5):
  t = 0

  steps = get_steps(S, D_over_d)
  try:
    step = next(steps)
  except StopIteration:
    return int(t/S)

  shift=0
  i=0

  while True:

    if i + step*(S-1) >= D_over_d:
      shift += 1

      if shift >= step or shift + step*(S-1) >= D_over_d:
        try:
          step = next(steps)
        except StopIteration:
          return int(t)
        shift = 0

      i = shift

    for s in range(S):
      i += step
    t += 1


def comp_in_sup_assignment(T = 2000, D_over_d = 500, S = 5):

  D_over_d = int(D_over_d)
  assignments = torch.zeros(T, D_over_d, dtype=torch.int64)
  compact_assignments = torch.zeros(T, S, dtype=torch.int64)

  if S == 1:
    i = 0
    for t in range(T):
      assignments[t,i] = 1
      compact_assignments[t,0] = i
      i += 1
      if i >= D_over_d:
        i = 0
    return assignments.to(device).float(), compact_assignments.to(device).int()

  steps = get_steps(S, D_over_d)
  step = next(steps)
  shift=0
  i=0

  for t in range(T):

    if i + step*(S-1) >= D_over_d:
      shift += 1

      if shift >= step or shift + step*(S-1) >= D_over_d:
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

#Parameters
D = 1000 # Number of neurons in the large network devided by 4
T = 6000 # Number of small circuits in superposition
S = 5 # Number of large network neurons used by each small circuit neuron

#Embedding assignments
assignments_1, compact_assignments_1 = comp_in_sup_assignment(T, D, S)
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
W1 = torch.zeros(4*D, 4*D, device=device)
W2 = torch.zeros(4*D, 4*D, device=device)

#Perserveing activationi indicators
W1[:2*D, :2*D] = torch.eye(2*D, device=device)
W2[:2*D, :2*D] = torch.eye(2*D, device=device)

#Adding 2 to active circuit neurons
W1[2*D:3*D, :D] = torch.eye(D, device=device) * 2
W1[3*D:4*D, :D] = torch.eye(D, device=device) * 2
W2[2*D:3*D, D:2*D] = torch.eye(D, device=device) * 2
W2[3*D:4*D, D:2*D] = torch.eye(D, device=device) * 2

#Removing rotated one
rotated_one = torch.einsum('tn,tm,tij,j->nmi', (assignments_1, assignments_2, r, one))/S
W1[2*D:3*D, D:2*D] = - rotated_one[:,:,0]
W1[3*D:4*D, D:2*D] = - rotated_one[:,:,1]
W2[2*D:3*D, :D] = - rotated_one[:,:,0].t()
W2[3*D:4*D, :D] = - rotated_one[:,:,1].t()

#Rotation
all_rotations= torch.einsum('tn,tm,tij->nmij', (assignments_1, assignments_2, r))/S
W1[2*D:3*D, 2*D:3*D] = all_rotations[:,:,0,0]
W1[2*D:3*D, 3*D:4*D] = all_rotations[:,:,0,1]
W1[3*D:4*D, 2*D:3*D] = all_rotations[:,:,1,0]
W1[3*D:4*D, 3*D:4*D] = all_rotations[:,:,1,1]
W2[2*D:3*D, 2*D:3*D] = all_rotations[:,:,0,0].t()
W2[2*D:3*D, 3*D:4*D] = all_rotations[:,:,0,1].t()
W2[3*D:4*D, 2*D:3*D] = all_rotations[:,:,1,0].t()
W2[3*D:4*D, 3*D:4*D] = all_rotations[:,:,1,1].t()

#Bias
B = torch.zeros(4*D, device=device)
B[2*D:] = 1


#%% Running the network
#   Running the network

#Parameters
L = 5 # Number of layers
z = 2 # Number of circuits in superposition

#Neuron activations
A = torch.zeros(L, 4*D, device=device)
x = torch.zeros(L, z, 2, device=device)
est_x = torch.zeros(L, z, 2, device=device)

#Input
active_circuits = torch.randint(T,(z,), device=device)
initial_angle = torch.rand(z, device=device) * 2 * np.pi
x[0,:,0] = torch.cos(initial_angle)
x[0,:,1] = torch.sin(initial_angle)
est_x[0] = x[0]

#Running the network
for l in range(1,L):
    x[l] = torch.einsum('tij,tj->ti', r[active_circuits], x[l-1])

    if l%2 == 1: # Odd layers
        if l == 1: # layer 1
            for k,t in enumerate(active_circuits):
                A[1, :D] += assignments_1[t]
                A[1, D:2*D] += assignments_2[t]
                A[1, 2*D:3*D] += (x[1,k,0] + 1) * assignments_1[t]  
                A[1, 3*D:4*D] += (x[1,k,1] + 1) * assignments_1[t] 
        else:
           A[l] = torch.relu(torch.einsum('ij,j->i', (W1, A[l-1])) - B)


        est_x[l,:,0] = torch.einsum('tn,n->t', 
                                    (assignments_1[active_circuits], A[l, 2*D:3*D]))/S - 1
        est_x[l,:,1] = torch.einsum('tn,n->t',  
                                    (assignments_1[active_circuits], A[l, 3*D:4*D]))/S - 1
        
    else: # Even layers
        A[l] = torch.relu(torch.einsum('ij,j->i', (W2, A[l-1])) - B)

        est_x[l,:,0] = torch.einsum('tn,n->t', 
                                    (assignments_2[active_circuits], A[l, 2*D:3*D]))/S - 1
        est_x[l,:,1] = torch.einsum('tn,n->t', 
                                    (assignments_2[active_circuits], A[l, 3*D:4*D]))/S - 1
#%% Print x and est_x
#   Print x and est_x
for k in range(z):
    for l in range(L):
        print(f'l={l}')
        print('x:    ', x[l,k])
        print('est_x:', est_x[l,k])
        print()
# %%
l=3
print(f'l={l}')
print(A[l,:D])
print(A[l,D:2*D])
print(A[l,2*D:3*D])
print(A[l,3*D:4*D])
# %%
