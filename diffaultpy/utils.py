import numpy as np
import torch
from scipy import integrate
from diffaultpy.crystal_structure import FCC_structure, BCC_structure, SC_structure


def exact_integral_for_wilkens(x):
    """Computes the integral of arcsin(y)/y from 0 to x using numerical integration
    Args:
        x (array): upper limit of the integral
    Returns:
        array: integral
    """
    toInt = lambda y: np.arcsin(y)/y # the integrand is arcsin(x)/x
    integrator = lambda y : integrate.quad(toInt, 0, y)[0]  # integrate from 0 to x
    return np.array([integrator(_) for _ in x]) # loop over the array and compute the integral

def get_crystal_structure(phase):
    if phase == 'fcc':
        return FCC_structure()
    elif phase == 'bcc':
        return BCC_structure()
    elif phase == 'sc':
        return SC_structure()
    else:
        raise ValueError("Unsupported phase type")
    
def convert_to_angle(x):
    """Converts an array of x values to angles
    Args:
        x (array): x values
    Returns:
        array: angles
    """
    return


def shift_each_column_numpy(array, shifts):
    """Shifts each column of the array by the corresponding value in shifts.
    Substitute the np.roll function with a custom implementation that works column-wise.

    Args:
        array (array): array to shift
        shifts (array): shifts, for each column
    Returns:
        array: shifted array
    """
    N, m = array.shape
    row_indices = (np.arange(N)[:, None] - shifts) % N # specify an index - array 
    return array[row_indices, np.arange(m)]
    

def shift_each_column_torch(array, shifts):
    """Shifts each column of the array by the corresponding value in shifts
    Args:
        array (array): array to shift
        shifts (array): shifts, for each column
    Returns:
        array: shifted array
    """
    N, m = array.shape
    row_indices = (torch.arange(N).unsqueeze(1) - shifts) % N
    return array[row_indices, torch.arange(m)]
