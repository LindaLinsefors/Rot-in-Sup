# Checking that the way chi is implicityly calcudlated 
# in the code matches the definition given in the post.
# 
# %%
# setup
import torch
from assignments import *
from classes_and_functions import *

# Create network

T = 557 # Number of circuits
D = 1000 # Neurons per layer in the large netowrk
d = 4 # Neurons per layer in the small circuits
Dod = D // d # D/d : This number is used a lot
S = 6 # Embeding redundancy: How many netowrk neurns are used to embed each circuit neuron.

L = 5 # Number of layers
z = 1 # Number of active circuits

circ = RotSmallCircuits_4d(T, b=1) # Create T small circuits
net = CompInSup(D, L, S, circ) # Create the large network
# 
# %%
# Checing chi calculations
# I.e. checking that the code match up to the definition in the post.

embed = net.embed
l=3

capped_embed = torch.einsum('tn,tm->nm',embed[l],embed[l-1])
capped_embed.clamp_(max=1.0)

inverted_capped_embed = torch.ones_like(capped_embed) - capped_embed

above_diag = torch.triu(torch.ones(T, T), diagonal=1).bool()

exp = torch.einsum('tn,nm,um->tu', embed[l], capped_embed, embed[l-1])[above_diag].mean()
inv_exp = torch.einsum('tn,nm,um->tu', embed[l], inverted_capped_embed, embed[l-1])[above_diag].mean()

chi = 1/S * exp/inv_exp

W = capped_embed/S - inverted_capped_embed*chi

new_exp = torch.einsum('tn,nm,um->tu', embed[l], W, embed[l-1])[above_diag].mean()

print(exp)
print(inv_exp)
print(new_exp)

every_unwanted_interaction = torch.einsum('tn,nm,um->tu', embed[l], capped_embed, embed[l-1])[above_diag].sum()
every_possible_interaction = T*(T-1)/2 * S*S
capped_corr_1 = every_unwanted_interaction/(every_possible_interaction-every_unwanted_interaction) #* 10

print(f'chi = {chi:.10f}')
print(f'capped_corr_1/S = {capped_corr_1/S:.10f}')