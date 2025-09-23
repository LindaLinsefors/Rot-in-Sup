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



    



# %% Very small test
#    Very small test



Dod=5
S=2
T=2
L=4
z=1
bs=1

for d in [3,4]:
    D=Dod*d
    print(f'd={d}')

    circ = RotSmallCircuits(T, 0.1, d)
    net = CompInSup(D, L, S, circ, u_correction=0)
    run = net.run(L, z, bs, active_circuits=torch.tensor([[0]]))

    if (run.x - run.est_x).sum().abs() > 1e-6 and (run.a - run.est_a).sum().abs() > 1e-6:
        print("CompInSup test failed: The network output does not match the circuit outputs.")
    elif not (run.x.abs() < 1).all():
        print("CompInSup test failed: The vector values are not in the expected range [-1, 1].")
    elif d == 3 and not (run.a[:,:,:,0] == 1).all():
        print("CompInSup test failed: The on-indicator values are not as expected (should be 1).")
    elif d == 4 and not (run.a[:,:,:,0] == 1.5).all() and not (run.a[:,:,:,1] == 0.5).all():
        print("CompInSup test failed: The on-indicator values are not as expected (should be 1.5 and 0.5).") 
    else:
        print("CompInSup test passed: The output matches the expected result.")



#%% Plot MSE #######################################################################################
#   Plot MSE


T = 500
D = 2*T
d = 4

Dod = D // d


bs = T
L = 5
S = 3

b = 1

f = frequency_of_overlap(T, Dod, S)
p = probability_of_overlap(T, Dod, S)


#u_correction = f/((S-f)*S)
#u_correction = p/((S-p)*S)
#u_correction = 1/(Dod-S)
#u_correction = 0
u_correction = None


runs = []
labels = []
expected = []

# for u_correction_type in [ 'p', 'f', 'D']:
#     if u_correction_type == 'p':
#         u_correction = p/((S-p)*S)
#     if u_correction_type == 'f':
#         u_correction = f/((S-f)*S)
#     if u_correction_type == 'D':
#         u_correction = 1/(Dod-S)

# for b in [0.3, 0.4, 0.5]:
#     for S in [3,4,5]:

#for u_correction in [0, 0.0021, 0.0022, 0.0023, 0.0024, 0.0025, 0.0026, 0.0027, 0.0028]:

#circ = RotSmallCircuits(T, b, d)
#net = CompInSup(D, L, S, circ, u_correction=u_correction)
#initial_angle = torch.rand(bs, z) * 2 * np.pi
#active_circuits = torch.randint(T, (bs, z))

#for z in [1, 2, 3]:

Ss = [5]

#nets = {}
#for S in Ss:
    #nets[S] = CompInSup(D, L, S, circ, u_correction=u_correction)

for z in [1,2]:    
    #for split, capped in [(False, False), (True, False), (True, True)]:
    for split, capped in [(True, True)]:
        for S in Ss:
            #for w_correction in [1, 1.5, 2]:
            for b in [0.0, 0.5, 1.0]:

                circ = RotSmallCircuits(T, b, d)
                net = CompInSup(D, L, S, circ)


                if (split, capped) == (False, False):
                    labels.append(f'z={z}, S={S}')
                if (split, capped) == (True, False):
                    labels.append(f'z={z}, split, S={S}')
                if (split, capped) == (True, True):
                    #labels.append(f'z={z}, capped, S={S}')
                    #labels.append(f'z={z}, w_corr={w_correction}')
                    labels.append(f'z={z}, b={b}')

                #net = CompInSup(D, L, S, circ, w_correction=w_correction)
                run = net.run(L, z, bs, 
                            #active_circuits=active_circuits, 
                            #initial_angle=initial_angle,
                            capped=capped, split=split)

                runs.append(run)
                #labels.append(f'corr type={u_correction_type}')
                #labels.append(f'b={b}, S={S}')
                #labels.append(f'z={z}, split={split}')
                #labels.append(f'corr={u_correction}')

                expected.append([expected_mse_rot(T,Dod,l,b,z) for l in range(L+1)]) 

# title = f'D={D}, D/d = {Dod}, T={T}, L={L}, z={z}, bs={bs}, S={S}, b={b}'
# title = f'D={D}, D/d = {Dod}, T={T}, L={L}, z={z}, bs={bs}, corr type = D'
title = f'D={D}, D/d = {Dod}, T={T}, L={L}, bs={bs}, S={S}, d={d}'

plot_mse_rot(L, labels, runs, title, expected)

plot_worst_error_rot(L, labels, runs, title)



#%%


T = 500
L = 5
b = 1

S = 6
D = 1000
d = 4
bs = T
z = 1
Dod = D // d
active_circuits = torch.arange(T).reshape(T, 1)


circ = RotSmallCircuits(T, b, d)
net = CompInSup(D, L, S, circ)
run = net.run(L, z, bs, capped=True, active_circuits=active_circuits)

#%%
plot_rot(run)
plot_mse_rot(L, [' '], [run], f'All circuits, D={D}, T={T}, L={L}, z={z}, S={S}, b={b}')
plot_worst_error_rot(L, [' '], [run], f'All circuits, D={D}, T={T}, L={L}, z={z}, S={S}, b={b}')
#%%

unemb = net.embed/S
A = run.A
a = run.a

est_a = torch.zeros(L+1, bs, T, d)

for l in range(L):
    for i in range(d):
        est_a[l+1, :, :, i] = torch.einsum('tn,bn->bt', unemb[l], A[l+1,:,i*Dod:(i+1)*Dod])

inactive_est_a = est_a.clone()
for l in range(1, L+1):
    for i in range(d):
        inactive_est_a[l,:,:,i].fill_diagonal_(0)

est_on = est_a[:,:,:,0] - est_a[:,:,:,1]
inactive_est_on = inactive_est_a[:,:,:,0] - inactive_est_a[:,:,:,1]


est_x = est_a[:,:,:,-2:] - est_on[:,:,:,None]
x = torch.zeros(L+1, bs, T, 2)
x[:,torch.arange(T),torch.arange(T),:] = run.x[:,:,0,:]

error_x = (est_x - x).norm(dim=-1)
inactive_error_x = error_x.clone()
inactive_error_x[:,torch.arange(T),torch.arange(T)] = 0
#%%

data = inactive_error_x[2]
L=2

plt.imshow(data, cmap='gray_r', aspect='equal')
plt.axis('off')
plt.colorbar()
plt.show()


#%%
l=2
worst_impact_on_others_on = inactive_est_on[l].max(dim=1).values
worst_impact_on_self_on   = inactive_est_on[l].max(dim=0).values
self_error_on                   = (est_on[l].diag() - 1).abs()

worst_impact_on_others_x = inactive_error_x[l].max(dim=1).values
worst_impact_on_self_x   = inactive_error_x[l].max(dim=0).values
self_error_x             = error_x[l].diag()

plt.plot(sorted(worst_impact_on_others_on, reverse=True), label='Worst impact on others, on')
plt.plot(sorted(worst_impact_on_self_on, reverse=True), label='Worst impact on self, on')
plt.plot(sorted(self_error_on, reverse=True), label='Self error, on')
plt.plot(sorted(worst_impact_on_others_x, reverse=True), label='Worst impact on others, x')
plt.plot(sorted(worst_impact_on_self_x, reverse=True), label='Worst impact on self, x')
plt.plot(sorted(self_error_x, reverse=True), label='Self error, x')
plt.grid(True)
plt.legend()
plt.show()

#%%

bs_2 = int(5*T)
z_2 = 2
run_2 = net.run(L, z_2, bs_2, capped=True)   

#%%

A = run_2.A
active_circuits = run_2.active_circuits

est_a_2 = torch.zeros(L+1, bs_2, T, d)
for l in range(L):
    for i in range(d):
        est_a_2[l+1, :, :, i] = torch.einsum('tn,bn->bt', unemb[l], A[l+1,:,i*Dod:(i+1)*Dod])

inactive_est_a_2 = est_a_2.clone()
inactive_est_a_2[:,torch.arange(bs_2).unsqueeze(1), active_circuits,:] = 0

inactive_est_on_2 = inactive_est_a_2[:,:,:,0] - inactive_est_a_2[:,:,:,1]
inactive_est_x_2 = inactive_est_a_2[:,:,:,-2:] - inactive_est_on_2[:,:,:,None]
inactive_error_x_2 = inactive_est_x_2.norm(dim=-1)

l=2
worst_impact_on_self_on_2 = inactive_est_on_2[l].max(dim=0).values
worst_impact_on_self_x_2 = inactive_error_x_2[l].max(dim=0).values

#%%

plt.plot(sorted(worst_impact_on_self_on_2, reverse=True), label='Worst impact on self, on, z=2')
plt.plot(sorted(worst_impact_on_self_x_2, reverse=True), label='Worst impact on self, x, z=2')
plt.grid(True)
plt.legend()
plt.title(f'z=2, T={T}, D={D}, d={d}, L={l}, S={S}')
plt.show()


#%%

plt.plot(sorted(worst_impact_on_others_on, reverse=True), label='Worst impact on others, on')
plt.plot(sorted(worst_impact_on_self_on, reverse=True), label='Worst impact on self, on')
plt.plot(sorted(worst_impact_on_self_on_2, reverse=True), label='Worst impact on self, on, z=2')
plt.plot(sorted(self_error_on, reverse=True), label='Self error, on')
plt.plot(sorted(worst_impact_on_others_x, reverse=True), label='Worst impact on others, x')
plt.plot(sorted(worst_impact_on_self_x, reverse=True), label='Worst impact on self, x')
plt.plot(sorted(worst_impact_on_self_x_2, reverse=True), label='Worst impact on self, x, z=2')
plt.plot(sorted(self_error_x, reverse=True), label='Self error, x')
plt.grid(True)
plt.title(f'T={T}, D={D}, d={d}, L={l}, S={S}')
plt.legend()
plt.show()

# %%


sort_by =  worst_impact_on_self_x_2
perm = torch.argsort(sort_by, descending=True)

#plt.plot(worst_impact_on_others_on[perm], label='Worst impact on others, on',alpha=0.5)
#plt.plot(worst_impact_on_self_on[perm], label='Worst impact on self, on',alpha=0.5)
#plt.plot(worst_impact_on_self_on_2[perm], label='Worst impact on self, on, z=2',alpha=0.5)
#plt.plot(self_error_on[perm], label='Self error, on',alpha=0.5)
plt.plot(worst_impact_on_others_x[perm], label='Worst impact on others, x',alpha=0.5)
plt.plot(worst_impact_on_self_x[perm], label='Worst impact on self, x',alpha=0.5)
plt.plot(worst_impact_on_self_x_2[perm], label='Worst impact on self, x, z=2',alpha=0.5)   
#plt.plot(self_error_x[perm], label='Self error, x',alpha=0.5)
plt.legend()
plt.show()

sort_by =  worst_impact_on_self_on_2
perm = torch.argsort(sort_by, descending=True)

plt.plot(worst_impact_on_others_on[perm], label='Worst impact on others, on',alpha=0.5)
plt.plot(worst_impact_on_self_on[perm], label='Worst impact on self, on',alpha=0.5)
plt.plot(worst_impact_on_self_on_2[perm], label='Worst impact on self, on, z=2',alpha=0.5)
#plt.plot(self_error_on[perm], label='Self error, on',alpha=0.5)
#plt.plot(worst_impact_on_others_x[perm], label='Worst impact on others, x',alpha=0.5)
#plt.plot(worst_impact_on_self_x[perm], label='Worst impact on self, x',alpha=0.5)
#plt.plot(worst_impact_on_self_x_2[perm], label='Worst impact on self, x, z=2',alpha=0.5)   
#plt.plot(self_error_x[perm], label='Self error, x',alpha=0.5)
plt.legend()
plt.show()

#%%
L=2

worst_impact_on_others_on = inactive_est_on[L].max(dim=0).values
worst_impact_on_self_on   = inactive_est_on[L].max(dim=1).values
self_error_on                   = (est_on[L].diag() - 1).abs()

worst_impact_on_others_x = inactive_error_x[L].max(dim=0).values
worst_impact_on_self_x   = inactive_error_x[L].max(dim=1).values
self_error_x             = error_x[L].diag()

plt.hist(worst_impact_on_others_on, bins=30, alpha=0.5 ,label='Worst impact on others, on')
plt.hist(worst_impact_on_self_on, bins=30, alpha=0.5 ,label='Worst impact on self, on')
plt.hist(self_error_on, bins=30, alpha=0.5 ,label='Self error, on')
plt.hist(worst_impact_on_others_x, bins=30, alpha=0.5 ,label='Worst impact on others, x')
plt.hist(worst_impact_on_self_x, bins=30, alpha=0.5 ,label='Worst impact on self, x')
plt.hist(self_error_x, bins=30, alpha=0.5 ,label='Self error, x')
plt.legend()
plt.show()

N=20
plt.plot(worst_impact_on_others_on[:N], 'o-', label='Worst impact on others, on')
plt.plot(worst_impact_on_self_on[:N], 'o-', label='Worst impact on self, on')
plt.plot(self_error_on[:N], 'o-', label='Self error, on')
plt.plot(worst_impact_on_others_x[:N], 'o-', label='Worst impact on others, x')
plt.plot(worst_impact_on_self_x[:N], 'o-', label='Worst impact on self, x')
plt.plot(self_error_x[:N], 'o-', label='Self error, x')
plt.legend()
plt.show()




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

u_correction = 1/(Dod-S)

circ = RotSmallCircuits(T, b)
net = CompInSup(D, L, S, circ, u_correction=u_correction)
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
u_correction = f/((S-f)*S)
#u_correction = 1/Dod

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
u_correction = p/((S-p)*S)
print(f"Correction: {u_correction:.4f}")

circ = RotSmallCircuits(T, b)
net = CompInSup(D, L, S, circ, u_correction=u_correction)
run = net.run(L, z, bs)
print(run.est_a[:,:,0,0].mean((-1)))

print(f"Frequency of overlap: {f:.4f}")
u_correction = f/((S-f)*S)
print(f"Correction: {u_correction:.4f}")

circ = RotSmallCircuits(T, b)
net = CompInSup(D, L, S, circ, u_correction=u_correction)
run = net.run(L, z, bs)
print(run.est_a[:,:,0,0].mean((-1)))

print("Correction = 1/(Dod-S)")
u_correction = 1/(Dod-S)
print(f"Correction: {u_correction:.4f}")
circ = RotSmallCircuits(T, b)
net = CompInSup(D, L, S, circ, u_correction=u_correction)
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

z = 1

f = frequency_of_overlap(T, Dod, S)
p = probability_of_overlap(T, Dod, S)


#u_correction = f/((S-f)*S)
#u_correction = p/((S-p)*S)
u_correction = 1/(Dod-S)


runs = []
labels = []
nets = []

capped = True
expected = None

circ = RotSmallCircuits(T, b)

for L in [2,3]:
    
    net = CompInSup(D, L, S, circ, u_correction=u_correction, capped=capped)
    run = net.run(L, z, bs)

    nets.append(net)
    runs.append(run)
    labels.append(f'L={L}')

title = ''

plot_mse(labels, runs, title, expected)
# %%


L=2
u_correction = 1/(Dod-S)

embed = torch.zeros(L, T, Dod)
unemb = torch.zeros(L, T, Dod)
assign = torch.zeros(L, T, S, dtype=torch.int64)

embed[0], assign[0] = comp_in_sup_assignment(T, Dod, S)
for l in range(1, L):
    shuffle = torch.randperm(T)
    embed[l] = embed[0][shuffle]
    assign[l] = assign[0][shuffle]

if u_correction is None:
    p = probability_of_overlap(T, Dod, S)
    u_correction = p/((S-p)*S)
unemb = - torch.ones(L, T, Dod) * u_correction
unemb += embed * (1/S + u_correction)

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

if u_correction is None:
    p = probability_of_overlap(T, Dod, S)
    u_correction = p/((S-p)*S)
unemb = - torch.ones(L, T, Dod) * u_correction
unemb += embed * (1/S + u_correction)

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
