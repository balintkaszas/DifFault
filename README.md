# DifFault

## Simulation of diffraction patterns of faulted crystals

This package performs the simulation of X-ray diffraction (XRD) patterns based on microstructural parameters and physical principles. These account for the effects of crystallite size, dislocations, and planar faults. 

The supported crystal structures are Body centered cubic (BCC), Face centered cubic (FCC), and Simple cubic (SC). After specifying the desired crystal structure to be modeled, given the microstructural parameters, Diffault computes the Fourier amplitudes of the profiles associated to crystallite size, dislocations and planar faults.  

The library is compatible with both Numpy and PyTorch backends for the numerical operations. 

For more information, we refer to the manuscript [1], submitted to SoftwareX. 

## Installation

1. Clone the repository with the command

```
git clone https://github.com/balintkaszas/diffault.git
```

2. Install the dependencies with pip

``` 
cd Diffault
pip install -r requirements.txt
```    
3. Install the package with 

```
pip install -e . 
```
    
## Basic usage 
After installation, the XRD patterns for an FCC crystal can be simulated by initializing a ```Peak``` object and specifying the microstructural parameters [2] for the function ```generate_multiple_peaks```. 

```python
from diffaultpy.peak_shapes import Peak, generate_multiple_peaks
import numpy as np 
import matplotlib.pyplot as plt

# simulate the first 5 peaks: h^2+k^2+l^2 < 14
kappa_max = np.sqrt(14) /  lattice_constant # in [1 / nm]
N_fourier = 8192
lattice_constant = 0.36 # in [nm]
burgers_vector = 0.255 # in [nm]
Ch00 = 0.36 # dimensionless
rho = 0.01 # in [1/nm^2]
m = 20 # in [nm]
Rstar = 5 # in [nm]
sigma = 0.1 # dimensionless
q = 3 # dimensionless
B = 0.05 # dimensionless
intensities = np.array([0.65, 0.25, 0.2, 0.16, 0.9]).reshape(5, 1)
single_peak = Peak(kappa_max,
                N_fourier, 
                Ch00,
                burgers_vector,
                lattice_constant)
spectrum = generate_multiple_peaks(single_peak,
                                m,
                                sigma,
                                rho,
                                Rstar,
                                q,
                                peak_intensities = intensities,
                                planar_fault_probability = B)
# visualize the generated spectrum 

plt.plot(single_peak.positive_diffraction_vectors, spectrum)
```




![image](docs/Sample_spectrum.jpg)

The ```examples/``` folder contains further examples and demonstrations of the usage of the library. These include 

- Discussion of the individual profiles, i.e., associated to size effects, planar faults, and dislocations. 
- Generating multiple XRD patterns simulateously with Numpy and PyTorch
- A basic example of X-ray Line Profile Analysis (XLPA) [2]. We use a simple regression routine implemented in SciPy to determine the microstructural parameters of an observed spectrum. 


## References

[1] B. Kaszás, P. Nagy, J. Gubicza, DifFault: simulation of diffraction patterns of faulted crystals, SoftwareX 27, 101860, 2024. 

[2] J. Gubicza, X-ray line profile analysis in materials science, IGI global,
2014.

## Bibtex 
The journal paper describing the software can be cited as 

```
@article{diffault,
title = {DifFault: Simulation of diffraction patterns of faulted crystals},
journal = {SoftwareX},
volume = {27},
pages = {101860},
year = {2024},
author = {Bálint Kaszás and Péter Nagy and Jenő Gubicza},
}
```
