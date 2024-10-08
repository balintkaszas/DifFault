import numpy as np
from scipy import special
from scipy import integrate
from diffaultpy.crystal_structure import FCC_structure, BCC_structure, SC_structure
import torch

class MathBackend:
    """MathBackend class that provides a unified interface for numpy and torch functions. 
    It is not intended to be used directly, but rather as a backend for other classes: only functionality that is directly required is implemented. """
    def __init__(self, backend='numpy', device='cpu', dtype=None):
        self.backend = backend
        if backend == 'torch':
            self.device = device
            if dtype is None:
                dtype = torch.float32
            self.dtype = dtype
            self.pi = torch.pi
        if backend == 'numpy':
            self.pi = np.pi
        if backend not in ['numpy', 'torch']:
            raise ValueError("Unsupported backend")

    def array(self, data, requires_grad=False):
        if self.backend == 'torch':
            return torch.tensor(data, dtype=self.dtype, device=self.device, requires_grad=requires_grad)
        else:
            return np.array(data)

    def linspace(self, start, stop, num):
        if self.backend == 'torch':
            return torch.linspace(start, stop, num, device = self.device)
        else:
            return np.linspace(start, stop, num)
    def sqrt(self, x):
        if self.backend == 'torch':
            return torch.sqrt(x)
        else:
            return np.sqrt(x)
            
    def arange(self, start, stop, step = 1):
        if self.backend == 'torch':
            return torch.arange(start, stop, step)
        else:
            return np.arange(start, stop, step)
        
    def ones(self, shape):
        if self.backend == 'torch':
            return torch.ones(shape, dtype=self.dtype, device=self.device)
        else:
            return np.ones(shape)
        
    def zeros(self, shape):
        if self.backend == 'torch':
            return torch.zeros(shape, dtype=self.dtype, device=self.device)
        else:
            return np.zeros(shape)
        

    def where(self, condition, x, y):
        if self.backend == 'torch':
            return torch.where(condition, x, y)
        else:
            return np.where(condition, x, y)
        
    def max(self, x, dim = None):
        if self.backend == 'torch':
            if dim is None:
                return torch.max(x)
            else:
                return torch.max(x, dim).values
        else:
            return np.max(x, dim)
        
    def min(self, x):
        if self.backend == 'torch':
            return torch.min(x)
        else:
            return np.min(x)
        
    def roll(self, x, shift, axis):
        if self.backend == 'torch':
            return torch.roll(x, shift, axis)
        else:
            return np.roll(x, shift, axis)
        
    def sum(self, x, axis=None):
        if self.backend == 'torch':
            return torch.sum(x, axis)
        else:
            return np.sum(x, axis)
        
    def real(self, x):
        if self.backend == 'torch':
            return x.real
        else:
            return np.real(x)
        

    def exp(self, x):
        if self.backend == 'torch':
            return torch.exp(x)
        else:
            return np.exp(x)
    

    def sqrt(self, x):
        if self.backend == 'torch':
            return torch.sqrt(x)
        else:
            return np.sqrt(x)
        
    def power(self, x, n):
        if self.backend == 'torch':
            return torch.pow(x, n)
        else:
            return np.power(x, n)
    def abs(self, x):
        if self.backend == 'torch':
            return torch.abs(x)
        else:
            return np.abs(x)
        
    def mod(self, x, y):
        if self.backend == 'torch':
            return torch.fmod(x, y)
        else:
            return np.fmod(x, y)
        
    def log(self, x):
        if self.backend == 'torch':
            return torch.log(x)
        else:
            return np.log(x)
        
    def erfc(self, x):
        if self.backend == 'torch':
            return torch.erfc(x)
        else:
            return special.erfc(x)
    
    def arcsin(self, x):
        if self.backend == 'torch':
            return torch.asin(x)
        else:
            return np.arcsin(x)
        
    def arctan(self, x):
        if self.backend == 'torch':
            return torch.atan(x)
        else:
            return np.arctan(x)
        
    def fftfreq(self, n, d=1.0):
        if self.backend == 'torch':
            return torch.fft.fftfreq(n, d=d)
        else:
            return np.fft.fftfreq(n, d=d)

    def fft(self, x, axis = None):
        if self.backend == 'torch':
            return torch.fft.fft(x)
        else:
            return np.fft.fft(x)

    def ifft(self, x, axis = -1):
        if self.backend == 'torch':
            return torch.fft.ifft(x, axis = axis)
        else:
            return np.fft.ifft(x, axis = axis)
        
    def fftshift(self, x, axes = None):
        if self.backend == 'torch':
            return torch.fft.fftshift(x, dim = axes)
        else:
            return np.fft.fftshift(x, axes = axes)


    def ifftshift(self, x, axes = None):
        if self.backend == 'torch':
            return torch.fft.ifftshift(x, dim = axes)
        else:
            return np.fft.ifftshift(x, axes = axes)
        

    def get_index_to_shift(self, s, s_to_shift):
        """Computes the index of the array s that is closest to the value stoshift. 
        Used to shift the array s to the value stoshift 
        """
        if self.backend == 'torch':
            distanceFrompeak = torch.abs(s - s_to_shift)
            index_to_shift = torch.argmin(distanceFrompeak, axis = 0)
            return index_to_shift
        else:
            distanceFrompeak = np.abs(s - s_to_shift)
            index_to_shift = np.argmin(distanceFrompeak, axis = 0)
            return index_to_shift

    def shift_each_column(self, x, shifts):
        if self.backend == 'torch':
            return shift_each_column_torch(x, shifts, device = self.device)
        else:
            return shift_each_column_numpy(x, shifts)
        
    
    def flip_each_column(self, x):
        """Flips each column of the array with torch.flip
        Args:
            x (array): array to flip
        Returns:
            array: flipped array
        """
        if self.backend == 'torch':
            return torch.flip(x, [0])
        else:
            return x[::-1, :]
        



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

def get_crystal_structure(phase, max_dimensionless_kappa):
    if phase == 'fcc':
        return FCC_structure(max_dimensionless_kappa)
    elif phase == 'bcc':
        return BCC_structure(max_dimensionless_kappa)
    elif phase == 'sc':
        return SC_structure(max_dimensionless_kappa)
    else:
        raise ValueError("Unsupported phase type")
    
def convert_to_angle(x, wave_length):
    """Converts an array of x values to angles according to the Bragg formula. 
    Args:
        x (array): x values
        wave_length (float): wave length of the x-rays
    Returns:
        array: diffraction angles
    """
    return np.arcsin(x * wave_length / 2)


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
    

def shift_each_column_torch(array, shifts, device = 'cpu'):
    """Shifts each column of the array by the corresponding value in shifts
    Args:
        array (array): array to shift
        shifts (array): shifts, for each column
    Returns:
        array: shifted array
    """
    N, m = array.shape
    row_indices = (torch.arange(N, device = device).unsqueeze(1) - shifts) % N
    return array[row_indices, torch.arange(m)]
