import numpy as np

def find_triples_for_sqrsum(N):
    triples = set()
    for h in range(N + 1):
        for k in range(h, N + 1):
            l_squared = N - h**2 - k**2
            if l_squared >= 0:
                l = int(np.sqrt(l_squared))
                if l >= k and h**2 + k**2 + l**2 == N:
                    triples.add((l, k, h))
    return np.array(list(triples))

def generate_miller_index_in_range(kappa_range):
    """ Generate miller indices in the given range of non-dimensional kappas. 

    """
    kappa_int = int(kappa_range)
    all_hkl = []
    for i in range(1, kappa_int+1):
        temp = find_triples_for_sqrsum(i)
        if temp.shape[0] > 0:
            all_hkl.extend(temp)
    return np.array(all_hkl)


class FCC_structure():
    """Contains the Miller-indices and subreflections for a Face Centered Cubic (FCC) crystal structure.
    """
    def __init__(self):
        self.miller_indices = [[1, 1, 1],
                              [2, 0, 0],
                              [2, 2, 0],
                              [3, 1, 1],
                              [2, 2, 2]]
        self.subreflections = {}
        
        self.subreflections['111'] = [[1, 1, 1],
                                      [-1, 1, 1],
                                      [1, -1, 1],
                                      [1,1,-1]]
        self.subreflections['200'] = [[2, 0, 0]]
        self.subreflections['220'] = [[2, 2, 0],
                                      [2, -2, 0]]
        self.subreflections['222'] = [[2, 2, 2],
                                      [-2, 2, 2],
                                     [2, -2, 2],
                                     [2, 2, -2]]
        self.subreflections['311'] = [[3, 1, 1], 
                                     [-3, 1, 1],
                                     [-3, -1, 1]]
        return 
    

class SC_structure():
    """Contains the Miller-indices for a Simple Cubic (SC) crystal structure.
    """
    def __init__(self, kappa_range):
        self.miller_indices = generate_miller_index_in_range(kappa_range)
        return 


class SC_structure_legacy():
    """Contains the Miller-indices for a Simple Cubic (SC) crystal structure.
    """
    def __init__(self):
        self.miller_indices = [[1, 0, 0],
                              [1, 1, 0],
                              [1, 1, 1],
                              [2, 0, 0],
                              [2, 1, 0],
                              [2, 1, 1],
                              [2, 2, 0]]
        return 


class BCC_structure():
    """Contains the Miller-indices for a Body Centered Cubic (BCC) crystal structure.
    """
    def __init__(self):
        self.miller_indices = [[1, 1, 0],
                              [2, 0, 0],
                              [2, 1, 1],
                              [2, 2, 0],
                              [3, 1, 0]]
        return 
