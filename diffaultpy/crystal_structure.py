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
