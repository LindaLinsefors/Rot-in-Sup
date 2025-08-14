#%%

from code import interact
import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm

device = 'cpu' 
torch.set_default_device(device)

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

from classes_and_functions import RunData, RotSmallCircuits, expected_mse



#%% Set up
#   Set up




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

        # for l in range(1,L):
        #     [[W[l,:Dod,     :Dod], W[l,:Dod,     Dod:2*Dod], W[l,:Dod,     2*Dod:]],
        #      [W[l,Dod:2*Dod,:Dod], W[l,Dod:2*Dod,Dod:2*Dod], W[l,Dod:2*Dod,2*Dod:]],
        #      [W[l,2*Dod:,   :Dod], W[l,2*Dod:,   Dod:2*Dod], W[l,2*Dod:,   2*Dod:]]
        #     ]= torch.einsum('tn,tij,tm->ijnm', unemb[l], w, embed[l-1])

        # W1 = W.clone()


        # Used for: not capped and split:
        W2 = torch.zeros(L, D, D)
        W2[0] = torch.eye(D)

        for l in range(1,L):
            temp_W = (torch.einsum('tn,ij,tm->ijnm', unemb[l], mean_w, embed[l-1])
                      +torch.einsum('tn,tij,tm->ijnm', embed[l]/S, diff_w, embed[l-1]))
            
            for i in range(d):
                for j in range(d):
                    W2[l, i*Dod:(i+1)*Dod, j*Dod:(j+1)*Dod] = temp_W[i,j]


        # for l in range(1,L):
        #     [[W[l,:Dod,     :Dod], W[l,:Dod,     Dod:2*Dod], W[l,:Dod,     2*Dod:]],
        #         [W[l,Dod:2*Dod,:Dod], W[l,Dod:2*Dod,Dod:2*Dod], W[l,Dod:2*Dod,2*Dod:]],
        #         [W[l,2*Dod:,   :Dod], W[l,2*Dod:,   Dod:2*Dod], W[l,2*Dod:,   2*Dod:]]
        #     ]= (torch.einsum('tn,ij,tm->ijnm', unemb[l], mean_w, embed[l-1])
        #         +torch.einsum('tn,tij,tm->ijnm', embed[l]/S, diff_w, embed[l-1]))
            
        # W2 = W.clone()

        
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

            # [[W[l,:Dod,     :Dod], W[l,:Dod,     Dod:2*Dod], W[l,:Dod,     2*Dod:]],
            #     [W[l,Dod:2*Dod,:Dod], W[l,Dod:2*Dod,Dod:2*Dod], W[l,Dod:2*Dod,2*Dod:]],
            #     [W[l,2*Dod:,   :Dod], W[l,2*Dod:,   Dod:2*Dod], W[l,2*Dod:,   2*Dod:]]
            # ]= (torch.einsum('tn,tij,tm->ijnm', embed[l], diff_w, embed[l-1])
            #     + torch.einsum('nm,ij->ijnm', capped_embed, mean_w)
            #     )/S
                
        # W3 = W.clone()

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
    


    
    

def plot_mse(labels, runs, title, expected=None, y_max=None):
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
            plt.plot([expected[i][l][0] for l in range(L+1)], 
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
            plt.plot([expected[i][l][1] for l in range(L+1)], 
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
    



# %% Very small test
#    Very small test

Dod=5
D=Dod*3#
S=2
T=2
L=4
z=1
bs=1

circ = RotSmallCircuits(T, 0.1)
net = CompInSup(D, L, S, circ, correction=0)
run = net.run(L, z, bs, active_circuits=torch.tensor([[0]]))

if (run.x - run.est_x).sum().abs() > 1e-6 and (run.a - run.est_a).sum().abs() > 1e-6:
    print("CompInSup test failed: The output does not match the expected result.")
else:
    print("CompInSup test passed: The output matches the expected result.")


#%% Plot MSE #######################################################################################
#   Plot MSE


D = 1200
T = 1000

S = 3
z = 3
bs = 800
L = 2
Dod = D // 3
S = 5
z = 1
b = 1

f = frequency_of_overlap(T, Dod, S)
p = probability_of_overlap(T, Dod, S)


#correction = f/((S-f)*S)
#correction = p/((S-p)*S)
correction = 1/(Dod-S)


runs = []
labels = []
expected = []

# for correction_type in [ 'p', 'f', 'D']:
#     if correction_type == 'p':
#         correction = p/((S-p)*S)
#     if correction_type == 'f':
#         correction = f/((S-f)*S)
#     if correction_type == 'D':
#         correction = 1/(Dod-S)

# for b in [0.3, 0.4, 0.5]:
#     for S in [3,4,5]:

#for correction in [0, 0.0021, 0.0022, 0.0023, 0.0024, 0.0025, 0.0026, 0.0027, 0.0028]:



circ = RotSmallCircuits(T, b)
net = CompInSup(D, L, S, circ, correction=correction)
#initial_angle = torch.rand(bs, z) * 2 * np.pi
#active_circuits = torch.randint(T, (bs, z))

#for z in [1, 2, 3]:
    
for split, capped in [(False, False), (True, False), (True, True)]:
    for z in [3, 2, 1]:

        if (split, capped) == (False, False):
            labels.append(f'z={z}')
        if (split, capped) == (True, False):
            labels.append(f'z={z}, split')
        if (split, capped) == (True, True):
            labels.append(f'z={z}, capped')

        net = CompInSup(D, L, S, circ, correction=correction)

        run = net.run(L, z, bs, 
                    #active_circuits=active_circuits, 
                    #initial_angle=initial_angle,
                    capped=capped, split=split)

        runs.append(run)
        #labels.append(f'corr type={correction_type}')
        #labels.append(f'b={b}, S={S}')
        #labels.append(f'z={z}, split={split}')
        #labels.append(f'corr={correction}')

        expected.append([expected_mse(T,Dod,l,b,z) for l in range(L+1)]) 

# title = f'D={D}, D/d = {Dod}, T={T}, L={L}, z={z}, bs={bs}, S={S}, b={b}'
# title = f'D={D}, D/d = {Dod}, T={T}, L={L}, z={z}, bs={bs}, corr type = D'
title = f'D={D}, D/d = {Dod}, T={T}, L={L}, bs={bs}, S={S}, b={b}'

plot_mse(labels, runs, title, expected)



#%%






















#%%
D = 1200
T = 1000

S = 5
z = 3
bs = 800
L = 5
Dod = D // 3
b = 0.5

correction = 1/(Dod-S)

circ = RotSmallCircuits(T, b)
net = CompInSup(D, L, S, circ, correction=correction)
run = net.run(L, z, bs)

embed = net.embed
active_circuits = run.active_circuits
pre_A = run.pre_A

for l in range(1, L):
    mask = torch.einsum('btn->bn', embed[l-1, active_circuits]) > 0

    act_pre_A_on = pre_A[l,:,:Dod][mask]
    ina_pre_A_on = pre_A[l,:,:Dod][~mask]

    act_pre_A_x = pre_A[l,:,Dod:2*Dod][mask]
    ina_pre_A_x = pre_A[l,:,Dod:2*Dod][~mask]

    act_pre_A_y = pre_A[l,:,2*Dod:3*Dod][mask]
    ina_pre_A_y = pre_A[l,:,2*Dod:3*Dod][~mask]

    plt.subplot(2, 1, 1)
    plt.title(f'Pre-Activation at layer {l}')
    plt.hist(act_pre_A_on.cpu().numpy(), bins=50, alpha=0.5, label='Active On', density=True)
    plt.hist(ina_pre_A_on.cpu().numpy(), bins=50, alpha=0.5, label='Inactive On', density=True)

    plt.xlim(-2,4)
    plt.axvline(x=0, color='k', linestyle='--')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.hist(act_pre_A_x.cpu().numpy(), bins=50, alpha=0.5, label='Active X', density=True)
    plt.hist(ina_pre_A_x.cpu().numpy(), bins=50, alpha=0.5, label='Inactive X', density=True)
    plt.hist(act_pre_A_y.cpu().numpy(), bins=50, alpha=0.5, label='Active Y', density=True)
    plt.hist(ina_pre_A_y.cpu().numpy(), bins=50, alpha=0.5, label='Inactive Y', density=True)

    plt.xlim(-2,4)
    plt.axvline(x=0, color='k', linestyle='--')
    plt.legend()

    plt.show()

#%%
mask = torch.zeros(bs,T, dtype=torch.bool)
for batch in range(bs):
    mask[batch, active_circuits[batch]] = True




#%%

Dod = 100
D = 300
T = 120
S = 5
z = 1
bs = 800
L = 10

p = probability_of_overlap(T, Dod, S)
f = frequency_of_overlap(T, Dod, S)

print(f"Probability of overlap: {p:.4f}")
print(f"Frequency of overlap: {f:.4f}")
#%%
correction = f/((S-f)*S)
#correction = 1/Dod

circ = RotSmallCircuits(T, 0.1)
net = CompInSup(D, L, S, circ)
run = net.run(L, z, bs)

x = run.x
est_x = run.est_x
a = run.a
est_a = run.est_a

print(est_a[:,:,0,0].mean((-1)))

# %%

D = 1200
T = 1000

# D = 300
# T = 120

S = 5
z = 1
bs = 800
L = 7
Dod = D // 3
b = 1

p = probability_of_overlap(T, Dod, S)
f = frequency_of_overlap(T, Dod, S)

print(f"Probability of overlap: {p:.4f}")
correction = p/((S-p)*S)
print(f"Correction: {correction:.4f}")

circ = RotSmallCircuits(T, b)
net = CompInSup(D, L, S, circ, correction=correction)
run = net.run(L, z, bs)
print(run.est_a[:,:,0,0].mean((-1)))

print(f"Frequency of overlap: {f:.4f}")
correction = f/((S-f)*S)
print(f"Correction: {correction:.4f}")

circ = RotSmallCircuits(T, b)
net = CompInSup(D, L, S, circ, correction=correction)
run = net.run(L, z, bs)
print(run.est_a[:,:,0,0].mean((-1)))

print("Correction = 1/(Dod-S)")
correction = 1/(Dod-S)
print(f"Correction: {correction:.4f}")
circ = RotSmallCircuits(T, b)
net = CompInSup(D, L, S, circ, correction=correction)
run = net.run(L, z, bs)
print(run.est_a[:,:,0,0].mean((-1)))

# %%

Dod = 100
D = 300
T = 120
S = 5

p = probability_of_overlap(T, Dod, S)
f = frequency_of_overlap(T, Dod, S)

corr_p = p/((S-p)*S)
corr_f = f/((S-f)*S)

embed, assign = comp_in_sup_assignment(T, Dod, S)

unemb_p = - torch.ones(T, Dod) * corr_p
unemb_p += embed * (1/S + corr_p)

unemb_f = - torch.ones(T, Dod) * corr_f
unemb_f += embed * (1/S + corr_f)

E_f = (unemb_f @ embed.T - torch.eye(T)).sum()/(T * (T - 1))
E_p = (unemb_p @ embed.T - torch.eye(T)).sum()/(T * (T - 1))
# %%

Dod = 100
D = 300
T = 120
S = 5

t=0
mask = torch.ones(T, dtype=torch.bool)
mask[t] = False

neighbour_neurons = torch.where((embed[mask].T @ embed[mask] @ embed[t])>0.5, 1, 0) - embed[t]
unemb_t = (embed[t] - 1/(S) * neighbour_neurons)/S



# %%


D = 1200
T = 1000

S = 5
z = 3
bs = 800
L = 2
Dod = D // 3
b = 1
S = 5
z = 1

f = frequency_of_overlap(T, Dod, S)
p = probability_of_overlap(T, Dod, S)


#correction = f/((S-f)*S)
#correction = p/((S-p)*S)
correction = 1/(Dod-S)


runs = []
labels = []
nets = []

capped = True
expected = None

circ = RotSmallCircuits(T, b)

for L in [2,3]:
    
    net = CompInSup(D, L, S, circ, correction=correction, capped=capped)
    run = net.run(L, z, bs)

    nets.append(net)
    runs.append(run)
    labels.append(f'L={L}')

title = ''

plot_mse(labels, runs, title, expected)
# %%


L=2
correction = 1/(Dod-S)

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

l=1

capped_embed = torch.einsum('tn,tm->nm',embed[l],embed[l-1])
capped_embed.clamp_(max=1.0)

above_diag = torch.triu(torch.ones(T, T), diagonal=1).bool()
every_unwanted_interaction = torch.einsum('tn,nm,um->tu', embed[l], capped_embed, embed[l-1])[above_diag].sum()
every_possible_interaction = T*(T-1)/2 * S*S
capped_corr_1 = every_unwanted_interaction/(every_possible_interaction-every_unwanted_interaction)

ces = capped_embed.sum()
capped_corr_2 = ces/(Dod**2-ces)

capped_embed -= (torch.ones_like(capped_embed) - capped_embed) * capped_corr_1

capped_re_embed = capped_embed/S

normal_re_embed = torch.einsum('tn,tm->nm',unemb[l],embed[l-1])



capped_outcome = torch.einsum('tn,nm,um', embed[l]/S, capped_re_embed, embed[l-1])
normal_outcome = torch.einsum('tn,nm,um', unemb[l], normal_re_embed, embed[l-1])

capped_noise = capped_outcome[above_diag]
normal_noise = normal_outcome[above_diag]

plt.hist(capped_noise, bins=30, alpha=0.5, label='Capped Noise, mean={:.5f}'.format(capped_noise.mean()), density=True)
plt.hist(normal_noise, bins=30, alpha=0.5, label='Normal Noise, mean={:.5f}'.format(normal_noise.mean()), density=True)
plt.legend()
plt.show()




# %%
plt.hist(capped_noise[capped_noise > 0.5], bins=30, alpha=0.5, label='Capped Noise')
plt.hist(normal_noise[normal_noise > 0.5], bins=30, alpha=0.5, label='Normal Noise')
plt.legend()
plt.show()

# %%



D = 1200
T = 1000

S = 5
z = 2
bs = 200//z
L = 6
Dod = D // 3
b = 0.0

circ = RotSmallCircuits(T, b)
net = CompInSup(D, L, S, circ, capped=True)
run = net.run(L, z, bs, capped=True)

est_a = run.est_a
est_a.shape

for i in range(z):
    for j in range(bs):
        plt.plot(est_a[:,j,i,0], alpha=0.1, color='red')

plt.grid(True)
plt.xlabel('Active Circuits On-Indicator')
plt.ylabel('Layer')
plt.title(f'D={D}, D/d = {Dod}, T={T}, L={L}, z={z}, bs={bs}, S={S}, b={b}')
#plt.show()



net = CompInSup(D, L, S, circ, capped=False)
run = net.run(L, z, bs, capped=False)

est_a = run.est_a
est_a.shape

for i in range(z):
    for j in range(bs):
        #plt.plot(est_a[:,j,i,0], alpha=0.1, color='blue')
        pass

plt.grid(True)

plt.show()
# %%








L = 2

w = circ.w
mean_w = circ.mean_w
diff_w = w - mean_w

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

for l in range(1,L):
    [[W[l,:Dod,     :Dod], W[l,:Dod,     Dod:2*Dod], W[l,:Dod,     2*Dod:]],
        [W[l,Dod:2*Dod,:Dod], W[l,Dod:2*Dod,Dod:2*Dod], W[l,Dod:2*Dod,2*Dod:]],
        [W[l,2*Dod:,   :Dod], W[l,2*Dod:,   Dod:2*Dod], W[l,2*Dod:,   2*Dod:]]
    ]= torch.einsum('tn,tij,tm->ijnm', unemb[l], w, embed[l-1])

W_no_split = W.clone()

for l in range(1,L):
    [[W[l,:Dod,     :Dod], W[l,:Dod,     Dod:2*Dod], W[l,:Dod,     2*Dod:]],
        [W[l,Dod:2*Dod,:Dod], W[l,Dod:2*Dod,Dod:2*Dod], W[l,Dod:2*Dod,2*Dod:]],
        [W[l,2*Dod:,   :Dod], W[l,2*Dod:,   Dod:2*Dod], W[l,2*Dod:,   2*Dod:]]
    ]= (torch.einsum('tn,ij,tm->ijnm', unemb[l], mean_w, embed[l-1])
        +torch.einsum('tn,tij,tm->ijnm', embed[l]/S, diff_w, embed[l-1]))
    

# %%
