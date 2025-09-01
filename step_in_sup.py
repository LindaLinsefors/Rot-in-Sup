# %% Setup step_in_sup
#    Setup step_in_sup


import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm

import importlib, classes_and_functions, assignments
importlib.reload(classes_and_functions)
importlib.reload(assignments)

from assignments import maxT

from classes_and_functions import (CompInSup, 
                                   random_active_circuits,
                                   get_inactive_circuits,
                                   expected_mse_rot)

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










# Plot for talk

D = 800
T = 1000

d = 2
Dod = D // d

L = 15
bs = 1800
n = 1

S = 5
z = 1

correction = None

circ = StepSmallCircuits(T)

power = 1


for is_active in [True, False]:
    plt.figure(figsize=(10, 5))

    for z in [2,1]:
        for S in [5,3]:
            for capped, correction in [(False, None), (True, 0)]:

                if capped:
                    label = f'z={z}, S={S}, capped W'
                elif correction is None:
                    label = f'z={z}, S={S}, balanced U'
                elif correction == 0:
                    label = f'z={z}, S={S}, no correction'
                else:
                    label = '?'

                for i in range(n):
                    net = CompInSup(D, L, S, circ, correction=correction)
                    run = net.run(L, z, bs, capped=capped)

                    if is_active:
                        mse = (run.est_a - run.a).pow(power).mean(dim=(1, 2)).pow(1/power)
                        est_x = run.est_a[:,:,:,0] - run.est_a[:,:,:,1]
                        mse_x = (est_x - 1).pow(power).mean(dim=(1, 2)).pow(1/power)
                    else:
                        active = run.active_circuits
                        inactive = get_inactive_circuits(active, T)
                        unemb = net.unemb
                        A = run.A
                        inactive_est_a = torch.zeros(L+1, bs, T-z, d)
                        for l in range(L):
                            for j in range(d):
                                inactive_est_a[l+1, :, :, j] = torch.einsum('btn,bn->bt', unemb[l, inactive], A[l+1,:,j*Dod:(j+1)*Dod])
                        mse = inactive_est_a.pow(power).mean(dim=(1, 2)).pow(1/power)
                        est_x = inactive_est_a[:,:,:,0] - inactive_est_a[:,:,:,1]
                        mse_x = (est_x).pow(power).mean(dim=(1, 2)).pow(1/power)

                    plt.subplot(1, 3, 1)
                    if i == 0:
                        line = plt.plot(mse[:, 0], label=label, marker='o')
                    else:
                        plt.plot(mse[:, 0], marker='o', color=line[0].get_color())

                    plt.subplot(1, 3, 2)
                    plt.plot(mse[:, 1], marker='o', color=line[0].get_color())  

                    plt.subplot(1, 3, 3)
                    plt.plot(mse_x, marker='o', color=line[0].get_color())

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.grid(True)
        plt.xlabel('Layer')
        if power == 1:
            plt.ylabel('Mean Error')
        elif power == 2:
            plt.ylabel('MSE')
        else:
            plt.ylabel(f'Normalised L{power} Norm for the Errors')
        if i == 0:
            plt.title('a_1')
            plt.legend()
        elif i == 1:
            plt.title('a_2')
        elif i == 2:
            plt.title('x = a_1 - a_2')
    if is_active:
        plt.suptitle(f'Active Error, D={D}, dT={2*T}, S={S}, batch size={bs}')
    else:
        plt.suptitle(f'Inactive Error, D={D}, dT={2*T}, S={S}, batch size={bs}')
    plt.tight_layout()
    plt.show()
#%%



plt.figure(figsize=(2.5,1.5))
y=[0,0,1,1]
x=[0,0.25,0.75,1]
plt.plot(x,y,'g')
plt.grid(True)
plt.xticks([0,0.25,0.5,0.75,1])
plt.yticks([0,0.25,0.5,0.75,1])
plt.show()




#%%

# Alt Plot for talk

D = 800
T = 1000

d = 2
Dod = D // d

L = 4
bs = 800
n = 1

S = 5
z = 1

correction = None

circ = StepSmallCircuits(T)

power = 10


for is_active in [True, False]:
    plt.figure(figsize=(10, 5))

    for z in [2,1]:
        for S in [5,3]:
            for capped, correction in [(False, None), (True, 0)]:
            
                if capped:
                    label = f'z={z}, S={S}, capped W'
                elif correction is None:
                    label = f'z={z}, S={S}, balanced U'
                elif correction == 0:
                    label = f'z={z}, S={S}, no correction'
                else:
                    label = '?'

                for i in range(n):
                    net = CompInSup(D, L, S, circ, correction=correction)
                    run = net.run(L, z, bs, capped=capped)

                    if is_active:
                        mse = (run.est_a - run.a).abs().max(1).values.max(1).values
                        est_x = run.est_a[:,:,:,0] - run.est_a[:,:,:,1]
                        mse_x = (est_x - 1).abs().max(1).values.max(1).values
                    else:
                        active = run.active_circuits
                        inactive = get_inactive_circuits(active, T)
                        unemb = net.unemb
                        A = run.A
                        inactive_est_a = torch.zeros(L+1, bs, T-z, d)
                        for l in range(L):
                            for j in range(d):
                                inactive_est_a[l+1, :, :, j] = torch.einsum('btn,bn->bt', unemb[l, inactive], A[l+1,:,j*Dod:(j+1)*Dod])
                        mse = inactive_est_a.abs().max(1).values.max(1).values
                        est_x = inactive_est_a[:,:,:,0] - inactive_est_a[:,:,:,1]
                        mse_x = (est_x).abs().max(1).values.max(1).values

                    plt.subplot(1, 3, 1)
                    if i == 0:
                        line = plt.plot(mse[:, 0], label=label, marker='o')
                    else:
                        plt.plot(mse[:, 0], marker='o', color=line[0].get_color())

                    plt.subplot(1, 3, 2)
                    plt.plot(mse[:, 1], marker='o', color=line[0].get_color())  

                    plt.subplot(1, 3, 3)
                    plt.plot(mse_x, marker='o', color=line[0].get_color())

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.grid(True)
        plt.xlabel('Layer')
        plt.ylabel(f'Max Norm Error')
        if i == 0:
            plt.title('a_1')
            plt.legend()
        elif i == 1:
            plt.title('a_2')
        elif i == 2:
            plt.title('x = a_1 - a_2')
    if is_active:
        plt.suptitle(f'Active Error, D={D}, dT={2*T}, S={S}, batch size={bs}')
    else:
        plt.suptitle(f'Inactive Error, D={D}, dT={2*T}, S={S}, batch size={bs}')
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
# Plot for talk

D = 800
T = 1000

S = 5
z = 2
bs = 8000
L = 6
Dod = D // 2
b = 0
capped = True
correction = None

n = 40
bins = [(i/(n-1)-0.5)*4 for i in range(n)] 
reduced_bs = 800


circ = StepSmallCircuits(T)
net = CompInSup(D, L, S, circ, correction=correction)
run = net.run(L, z, bs, capped=capped)



fig = plt.figure(figsize=(8, 3))

active_error = run.est_a - run.a
for l in [2,4]:

    for i in range(3):
        plt.subplot(1, 3, i+1)
        if i==2:
            plt.title('x = a_0 - a_1')
            error = active_error[:,:,:,0] - active_error[:,:,:,1]
        else:
            plt.title(f'a_{i}')
            error = active_error[:,:,:,i]

        mse = error[l].pow(2).mean().item()
        plt.hist(error[l].flatten().cpu().numpy(), bins=bins, density=True, alpha=0.5, label=f'layer {l}, MSE={mse:.4f}')

        # Normal curve
        x = np.linspace(-0.4, 0.4, 200)
        variance = (expected_mse_rot(T, Dod, l, b, z)[1]/2)
        pdf = norm.pdf(x, loc=0, scale=variance**0.5)
        #plt.plot(x, pdf, label=f'Normal, sigma^2={variance:.4f}')

        plt.legend()

plt.subplot(1,3,1)
if capped:
    fig.suptitle(f'Error distribution for active circuits. Modified W.\n D={D}, dT={2*T}, S={S}, z={z}, batch size={bs}')  
else:
    fig.suptitle(f'Error distribution for active circuits. Modified U.\n D={D}, dT={2*T}, S={S}, z={z}, batch size={bs}')      
plt.tight_layout()
plt.show()




active = run.active_circuits[:reduced_bs]
inactive = get_inactive_circuits(active, T)
unemb = net.unemb
A = run.A[:, :reduced_bs]

inactive_error = torch.zeros(L+1, reduced_bs, T-z, 2)

for l in range(L):
    inactive_error[l+1, :, :, 0] = torch.einsum('btn,bn->bt', unemb[l, inactive], A[l+1, :, :Dod]) 
    inactive_error[l+1, :, :, 1] = torch.einsum('btn,bn->bt', unemb[l, inactive], A[l+1, :, Dod:])


fig = plt.figure(figsize=(8, 3))

for l in [2,4]:

    for i in range(3):
        plt.subplot(1, 3, i+1)
        if i==2:
            plt.title('x = a_0 - a_1')
            error = inactive_error[:,:,:,0] - inactive_error[:,:,:,1]
        else:
            plt.title(f'a_{i}')
            error = inactive_error[:,:,:,i]

        mse = error[l].pow(2).mean().item()
        plt.hist(error[l].flatten().cpu().numpy(), bins=bins, density=True, alpha=0.5, label=f'layer {l}, MSE={mse:.4f}')

        # Normal curve
        x = np.linspace(-0.4, 0.4, 200)
        variance = (expected_mse_rot(T, Dod, l, b, z)[1]/2)
        pdf = norm.pdf(x, loc=0, scale=variance**0.5)
        #plt.plot(x, pdf, label=f'Normal, sigma^2={variance:.4f}')

        plt.legend()

plt.subplot(1,3,1)
if capped:
    fig.suptitle(f'Error distribution for inactive circuits. Modified W.\n D={D}, dT={2*T}, S={S}, z={z}, batch size={reduced_bs}')  
else:
    fig.suptitle(f'Error distribution for inactive circuits. Modified U.\n D={D}, dT={2*T}, S={S}, z={z}, batch size={reduced_bs}')      
plt.tight_layout()
plt.show()

#%%















for l in [2,4]:

    mse = inactive_error[l].pow(2).mean().item()
    plt.hist(inactive_error[l].flatten().cpu().numpy(), bins=100, density=True, alpha=0.5, label=f'layer {l}, MSE={mse:.4f}')

    # Normal curve
    x = np.linspace(-0.5, 0.5, 200)
    variance = (expected_mse_rot(T, Dod, 1, b, z+1)[1]/2)
    pdf = norm.pdf(x, loc=0, scale=variance**0.5)
    plt.plot(x, pdf, label=f'Normal, sigma^2={variance:.4f}')

plt.legend()
plt.title(f'Error distribution for inactive circuits\n D={D}, dT={2*T}, S={S}, z={z}, batch size={bs}')    
plt.show()


for l in [2,4]:

    mse = inactive_error[l].pow(2).mean().item()
    plt.hist(inactive_error[l].flatten().cpu().numpy(), bins=50, density=True, alpha=0.5, label=f'layer {l}, MSE={mse:.4f}')

    # Normal curve
    x = np.linspace(-0.5, 0.5, 200)
    variance = (expected_mse_rot(T, Dod, 1, b, z+1)[1]/2)
    pdf = norm.pdf(x, loc=0, scale=variance**0.5)
    plt.plot(x, pdf, label=f'Normal, sigma^2={variance:.4f}')

plt.ylim(0, 0.5)
plt.legend()
plt.title(f'Error distribution for inactive circuits\n D={D}, dT={2*T}, S={S}, z={z}, batch size={reduced_bs}')    
plt.show()
# %%


for l in [2,4]:

    mse = inactive_error[l].pow(2).mean().item()
    plt.hist(inactive_error[l].flatten().cpu().numpy(), bins=100, density=True, alpha=0.5, label=f'layer {l}, MSE={mse:.4f}')

    # Normal curve
    x = np.linspace(-0.3, 0.3, 200)
    variance = (expected_mse_rot(T, Dod, 1, b, z+1)[1]/2)
    pdf = norm.pdf(x, loc=0, scale=variance**0.5)
    plt.plot(x, pdf, label=f'Normal, sigma^2={variance:.4f}')

plt.ylim(0, 0.0001)
plt.legend()
plt.title(f'Error distribution for inactive circuits\n D={D}, dT={2*T}, S={S}, z={z}, batch size={reduced_bs}')    
plt.show()
# %%
