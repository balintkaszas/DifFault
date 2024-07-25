import diffaultpy.crystal_structure as crystal_structure
import numpy as np

def test_FCC_structure():
    fcc = crystal_structure.FCC_structure()
    assert fcc.miller_indices == [[1, 1, 1],
                                  [2, 0, 0],
                                  [2, 2, 0],
                                  [3, 1, 1],
                                  [2, 2, 2]]
    assert fcc.subreflections['111'] == [[1, 1, 1],
                                      [-1, 1, 1],
                                      [1, -1, 1],
                                      [1,1,-1]]
    assert fcc.subreflections['200'] == [[2, 0, 0]]
    assert fcc.subreflections['220'] == [[2, 2, 0],
                                      [2, -2, 0]]
    assert fcc.subreflections['222'] == [[2, 2, 2],
                                      [-2, 2, 2],
                                     [2, -2, 2],
                                     [2, 2, -2]]
    assert fcc.subreflections['311'] == [[3, 1, 1], 
                                     [-3, 1, 1],
                                     [-3, -1, 1]]
    return

def test_BCC_structure():
    bcc = crystal_structure.BCC_structure(12)
    assert bcc.miller_indices == [[1, 1, 0],
                                  [2, 0, 0],
                                  [2, 1, 1],
                                  [2, 2, 0],
                                  [3, 1, 0]]
    return

def test_generate_miller_index_in_range():
    hkl_sc = crystal_structure.generate_miller_index_in_range(8)
    sc = crystal_structure.SC_structure(8)
    assert np.allclose(np.array(sc.miller_indices), hkl_sc)

if __name__ == "__main__":
    test_generate_miller_index_in_range()