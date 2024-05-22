# Diffault

## Simulation of diffraction patterns of faulted crystals

This package performs the simulation of X-ray diffraction (XRD) patterns based on microstructural parameters and physical principles. These account for the effects of crystallite size, dislocations, and planar faults. 

The supported crystal structures are Body centered cubic (BCC), Face centered cubic (FCC), and Simple cubic (SC). After specifying the desired crystal structure to be modeled, given the microstructural parameters, Diffault computes the Fourier amplitudes of the profiles associated to crystallite size, dislocations and planar faults. 

The library supports both Numpy and PyTorch backends for the numerical operations.


```
import diffaultpy as dp
kappa_max = 14
N_fourier = 8192
lattice_constant = 0.36
burgers_vector = 0.255
Ch00 = 0.36
single_peak = dp.Peak(kappa_max, N_fourier, Ch00, burgers_vector, lattice_constant
spectrum = dp.generate_multiple_peaks(single_peak, m, sigma, rho, Rstar, q, peak_intensities = intensities, planar_fault_probability = planar_fault_probability)
```

For more information on the usage and impact 