import numpy as np
import itertools

def find_triples_for_sqrsum(N):
    """Generate all indices h, k, l such that h^2 + k^2 + l^2 = N
    Different permutations are not counted: The indices are ordered according to h>=k>=l"""
    triples = set()
    for h in range(N + 1):
        for k in range(h, N + 1):
            l_squared = N - h**2 - k**2
            if l_squared >= 0:
                l = int(np.sqrt(l_squared))
                if l >= k and h**2 + k**2 + l**2 == N:
                    triples.add((l, k, h))
    return np.array(list(triples))

def generate_indices_in_range(kappa_sq_a_range):
    """ Generate indices, h, k, l in the given range of non-dimensional kappas. 
    These reflections are all present in a simple cubic lattice. 
    """
    kappa_int = int(kappa_sq_a_range)
    all_hkl = []
    for i in range(1, kappa_int+1):
        temp = find_triples_for_sqrsum(i)
        if temp.shape[0] > 0:
            all_hkl.extend(temp)
    return np.array(all_hkl)



def is_reflection_observed_fcc(h, k, l):
    """Check whether the given reflection h,k,l is present in an FCC lattice.
    For an FCC lattice, we only see a reflection if either 
    - h, k, l  are all even
    - h, k, l are all odd
    """
    if h%2 == 0 and k%2 == 0 and l%2 == 0:
        return True
    
    elif h%2 == 1 and k%2 == 1 and l%2 == 1:
        return True
    else:
        return False


def is_reflection_observed_bcc(h, k, l):
    """Check whether the given reflection h,k,l is present in a BCC lattice.
    For a BCC lattice, we only see a reflection if 
    - h + k + l is even
    """
    if (h + k + l)%2 == 0:
        return True
    else:
        return False


class FCC_structure():
    """Contains the indices and subreflections for a Face Centered Cubic (FCC) crystal structure.
    """
    def __init__(self, kappa_sq_a_range):
        
        self.reflection_indices = self.remove_systematic_abscences(
            generate_indices_in_range(kappa_sq_a_range))
        return 
    def remove_systematic_abscences(self, index):
        n_peaks = index.shape[0]
        surviving_index = [] 
        for n in range(n_peaks ):
            h, k, l = index[n, :]
            if is_reflection_observed_fcc(h, k, l):
                surviving_index.append([h, k, l])
        return np.array(surviving_index)
    
    def generate_subreflections(self, h, k, l):
        """Generate all subreflections obtained as 
        - all permutations of h, k, l 
        - all sign flips of one of the indices """
        permutations = set(itertools.permutations([h, k, l]))
        
        sign_flips = set()
        for perm in permutations:
            sign_flips.add((perm[0], perm[1], perm[2]))
            sign_flips.add((-perm[0], perm[1], perm[2]))
            sign_flips.add((perm[0], -perm[1], perm[2]))
            sign_flips.add((perm[0], perm[1], -perm[2]))
        
        result = np.array(list(sign_flips))
        
        return result

    
class SC_structure():
    """Contains the indices for a Simple Cubic (SC) crystal structure.
    """
    def __init__(self, kappa_sq_a_range):
        self.reflection_indices = generate_indices_in_range(kappa_sq_a_range)
        return 



class BCC_structure():
    """Contains the indices for a Body Centered Cubic (BCC) crystal structure.
    """
    def __init__(self, kappa_sq_a_range):
        
        self.reflection_indices = self.remove_systematic_abscences(
            generate_indices_in_range(kappa_sq_a_range))
        return 
    def remove_systematic_abscences(self, index):
        n_peaks = index.shape[0]
        surviving_index = [] 
        for n in range(n_peaks ):
            h, k, l = index[n, :]
            if is_reflection_observed_bcc(h, k, l):
                surviving_index.append([h, k, l])
        return np.array(surviving_index)


