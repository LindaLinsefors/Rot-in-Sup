"""
Assignments for circuits in a superposition

This module contains functions for generating and testing assignments
for circuits in superposition networks.
"""

import torch
from sympy.ntheory import isprime


def get_steps(S, Dod):
    """Generate valid step sizes for circuit assignments.
    
    Args:
        S: Number of large network neurons used by each small circuit neuron
        Dod: Number of neurons in the large network divided by 4
        
    Yields:
        Valid step sizes for the assignment algorithm
    """
    primes_smaller_than_S = [num for num in range(2, S) if isprime(num)]

    step = 1
    while step * (S-1) < Dod:
        for p in primes_smaller_than_S:
            if step % p == 0:
                break
        else:
            yield step
            if not isprime(S):
                n = 1
                while step * S**n * (S-1) < Dod:
                    yield step * S**n
                    n += 1
        step += 1


class MaxT(Exception):
    """Exception raised when maximum number of circuits T is exceeded."""
    pass


def maxT(Dod=500, S=5):
    """Calculate the maximum number of circuits that can be assigned.
    
    Args:
        Dod: Number of neurons in the large network divided by 4
        S: Number of large network neurons used by each small circuit neuron
        
    Returns:
        Maximum number of circuits T
    """
    t = 0
    
    steps = get_steps(S, Dod)
    try:
        step = next(steps)
    except StopIteration:
        return int(t/S)

    shift = 0
    i = 0

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


def comp_in_sup_assignment(T=2000, Dod=500, S=5, device="cpu"):
    """Compute assignments for circuits in superposition.
    
    Args:
        T: Number of small circuits in superposition
        Dod: Number of neurons in the large network divided by 4
        S: Number of large network neurons used by each small circuit neuron
        device: PyTorch device ("cpu" or "cuda")
        
    Returns:
        Tuple of (assignments, compact_assignments):
        - assignments: Tensor of shape (T, Dod) with assignment matrix
        - compact_assignments: Tensor of shape (T, S) with compact representation
    """
    Dod = int(Dod)
    assignments = torch.zeros(T, Dod, dtype=torch.int64)
    compact_assignments = torch.zeros(T, S, dtype=torch.int64)

    if S == 1:
        i = 0
        for t in range(T):
            assignments[t, i] = 1
            compact_assignments[t, 0] = i
            i += 1
            if i >= Dod:
                i = 0
        return assignments.to(device).float(), compact_assignments.to(device).int()

    steps = get_steps(S, Dod)
    step = next(steps)
    shift = 0
    i = 0

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
            assignments[t, i] = 1
            compact_assignments[t, s] = i
            i += step

    return assignments.to(device).float(), compact_assignments.to(device).int()


def test_assignments(assignments, S, device="cpu"):
    """Test the validity of circuit assignments.
    
    Args:
        assignments: Assignment matrix of shape (T, Dod)
        S: Number of large network neurons used by each small circuit neuron
        device: PyTorch device for computations
    """
    T = assignments.shape[0]
    
    not_S = (assignments.sum(dim=1) != S).sum()
    if not_S == 0:
        print('Test 1 passed: Correct number of assignments for all circuits')
    else:
        print(f'Test 1 failed: Wrong number of assignments for {not_S} circuit(s)')

    overlap = (assignments.to(torch.float)) @ (assignments.to(torch.float).T) - S * torch.eye(T, device=device)
    if overlap.max() > 1:
        print('Test 2 failed: Overlap is above one, some pair of circuits')
    else:
        print('Test 2 passed: Overlap is max one, for all pairs of circuits')


def slow_test_assignments(assignments, S):
    """Slow but thorough test of assignment validity.
    
    Args:
        assignments: Assignment matrix of shape (T, Dod)
        S: Number of large network neurons used by each small circuit neuron
        
    Returns:
        True if all tests pass, False otherwise
    """
    T = assignments.shape[0]

    passed_test_1 = True
    for t in range(T):
        if not assignments[t].sum() == S:
            print('Error: Wrong number of assignments for circuit', t)
            passed_test_1 = False

    if passed_test_1:
        print('Test passed: Correct number of assignments for all circuits')

    passed_test_2 = True
    for t in range(T):
        for u in range(t):
            if (assignments[t] * assignments[u]).sum() > 1:
                print('Error: Too high collision for circuits pair', u, t)
                passed_test_2 = False

    if passed_test_2:
        print('Test passed: Overlap is max one, for all pairs of circuits')

    if passed_test_1 and passed_test_2:
        return True
    else:
        return False

def expected_overlap_error(T, Dod, S, naive=False):
    if Dod >= T * S:
        return 0
    elif naive:
        return S/Dod
    else:
        return (T * S/Dod - 1) / (T - 1)
    
def probability_of_overlap(T, Dod, S, naive=False): 
    return expected_overlap_error(T, Dod, S, naive=naive) * S

def expected_squared_overlap_error(T, Dod, S, naive=False):
    return expected_overlap_error(T, Dod, S, naive=naive) / S


def frequency_of_overlap(T, Dod, S):
    assignments, _ = comp_in_sup_assignment(T, Dod, S)
    overlap = (assignments.to(torch.float)) @ (assignments.to(torch.float).T) - S * torch.eye(T)    
    overlap = overlap[overlap > 0.5]
    return overlap.numel() / (T * (T - 1))


class Test():
    pass