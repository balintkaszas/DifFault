
import numpy as np 
from diffaultpy.math_backend import MathBackend
from diffaultpy.wilkens_function import WilkensFunction

from diffaultpy.utils import get_crystal_structure
class Peak():
    '''
    General class containing the parameters and functions to characterize a single peak. 
    The peak is characterized by the Fourier coefficients of the convolution of the dislocation, size and stacking fault profiles.
    '''
    
    def __init__(self, 
                 max_range_diffraction_vector,
                max_fourier_components, 
                Ch00,
                burgers_vector,
                lattice_constant,
                backend = 'numpy',
                 device='cpu', dtype=None,
                phase = 'fcc', 
                approximation_wilkens = 'polynomial',
                stacking_or_twin_fault = 'stacking',
                use_reduced_rho = False,
                minimum_fwhm = 1e-2):
        """Initialize the peak from parameter, the vector of wave-numbers and the phase of the crystal structure.
        An approximation method might be specified for evaluation of the Wilkens function. 

        Args:
            Ch00 (float): Contrast factor for the (h00) reflection
            burgers_vector (float): Burgers vector
            lattice_constant (float): Lattice constant
            backend (string): Can be 'numpy' or 'torch'. Defaults to 'numpy'. All 
            math operations are done with this backend,e.g. 
                                                        numpy.linspace() or torch.linspace().
            device (string): Device for the torch backend. Defaults to 'cpu'.
            dtype (string): Data type for the torch backend. Defaults to float.
            max_range_diffraction_vector (float): Maxmimal allowed value for the norm of the diffraction vector
            approximation_wilkens (string): Can be 'exact', 'polynomial' or 'interpolation'. Defaults to 'exact'.
            If 'polynomial', the integral in the Wilkens function is approximated by a 5th order polynomial. 
            stacking_or_twin_fault (string): Expected type of planar faults. Can be 'stacking' or 'twin' Defaults to 'stacking'.    
            max_fourier_components (int): Number of points in the Fourier transform
            phase (str, optional): Phase of the crystal structure. Defaults to 'fcc'.
            use_reduced_rho (bool, optional): If True, the dislocation density is normalized by the contrast factor and the burgers vector.
                In this case, the profiles are universal since they only depend on rhostar = rho * b^2 * C_h00. Defaults to False.
            minimum_fwhm (float, optional): Minimum value of the Full Width at Half Maximum, if |h+k+l| is divisible by 3.
              Defaults to 1e-2.
        """

        self.max_range_diffraction_vector = max_range_diffraction_vector 
        self.C_h00 = Ch00
        self.burgers_vector = burgers_vector
        self.lattice_constant = lattice_constant
        self.Nfourier = max_fourier_components
        self.phase = phase
        self.planar_fault_type = stacking_or_twin_fault
        self.minimum_fwhm = minimum_fwhm
        self.math = MathBackend(backend, device, dtype)
        self.structure = get_crystal_structure(self.phase)
        self.use_reduced_rho = use_reduced_rho
        self.diffraction_vectors = self.math.linspace(-self.max_range_diffraction_vector,
                                                self.max_range_diffraction_vector,
                                                  self.Nfourier)
        self.positive_diffraction_vectors = self.diffraction_vectors[self.diffraction_vectors >= 0]
        self.wilkens_function = WilkensFunction(backend, approximation_wilkens, device, dtype)
        self.wilkens_function.initialize_with_method()
        return 
        
    def _H(self, h, k, l):
        """Computes the H function, a quartic combination of the Miller indices."""
        sq = h**2 + k**2 + l**2
        return (h**2 * k**2 + k**2 * l**2 + h**2 * l**2) / sq**2
    
    def _C_hkl_without_00(self, h, k, l, q):
        """Computes the contrast factor for the Miller indices and the q parameter. C_h00 is not included in the calculation."""
        return (1. - q * self._H(h,k,l))


    def _full_width_half_max(self, a, h, k, l, planar_fault_probability):
        """Calculate the Full Width at Half Maximum (FWHM) of the peak for a given set of Miller indices and the fault probability parameter.
        If the sum of the Miller indices is divisible by 3, the FWHM is set to a minimum value. 
        Args:
            a (float): Lattice constant
            Miller indices: h,k,l must be integers 
            planar_fault_probability (float): Can be array as well. Probability of planar faults: either twin or stacking. 
        Returns:
            float: FWHM of the peak
        """
        sqrsum = h**2 + k**2 + l**2
        abssum = self.math.array(int(self.math.abs(h + k + l)))
        
        if self.planar_fault_type == 'twin':
            factor_due_to_planar_fault = 2. * planar_fault_probability / self.math.sqrt(1. - planar_fault_probability)
        elif self.planar_fault_type == 'stacking':
            factor_due_to_planar_fault = self.math.log(1. / (1. - 3 * planar_fault_probability + 3.* planar_fault_probability**2 ))
        if self.math.mod(abssum, self.math.array(3)) == 0: 
            return self.minimum_fwhm
        else:
            return (1 / (3 * a )) * ( self.math.abs(h + k + l) / self.math.sqrt( sqrsum ) ) * factor_due_to_planar_fault
        
    def _delta_hkl(self, a, h, k, l, planar_fault_probability, intrinsic_or_extrinsic = 'intrinsic'):
        """Computes the delta function for a given set of Miller indices and the probability of the planar faults. delta > 0 only for stacking faults. 
        Args:
            h (int): Miller index
            k (int): Miller index
            l (int): Miller index
            planar_fault_probability (float): Probability of planar faults.  
            intrinsic_or_extrinsic (string, optional): Can be 'intrinsic' or 'extrinsic'. Defaults to 'intrinsic'.
        Returns:
            float: shift due to stacking faults
        """
        prefactor = 1 
        if intrinsic_or_extrinsic == 'extrinsic':
            prefactor = -1
        sqrsum = h**2 + k**2 + l**2
        abssum = int(self.math.abs(h + k + l))
        # here np.sqrt(3) is always a scalar (no torch.sqrt)

        if self.planar_fault_type == 'twin':
            prefactor = 0. # there is no shift due to twin faults
        factor_due_to_planar_fault = self.math.arctan(np.sqrt(3) * (1 - 2 * planar_fault_probability)) - self.math.pi / 3
        return prefactor * (1 / (3 * a )) * ( abssum / self.math.sqrt( sqrsum ) ) * factor_due_to_planar_fault

    def fourier_coefficients_planar_fault(self, L, planar_fault_probability, h, k, l, intrinsic_or_extrinsic = 'intrinsic'):
        """Calculates the Fourier coefficients for the stacking fault or twin fault profile for a given set of wave numbers, planar fault probability and Miller indices.
        Args:
            L (array): Wave numbers
            planar_fault_probability (float or array): B, or alpha parameters. 
            h (int): Miller index
            k (int): Miller index
            l (int): Miller index
        Returns:
            array: Fourier transform of the crystal structure
        """
        return self.math.exp(-2*self.math.pi*((self._full_width_half_max(self.lattice_constant, h, k, l, planar_fault_probability) / 2) * self.math.abs(L) 
                                - L * 1j* self._delta_hkl(self.lattice_constant, h, k, l, planar_fault_probability, intrinsic_or_extrinsic)))

    def fourier_coefficients_size(self, L, m, sigma):
        """Calculate the Fourier transform of the profile attributed to the size-effect for a given set of wave numbers, median and standard deviation. The size distribution is assumed to be log-normal.

        Args:
            L (array): Wave numbers
            m (float or array): Median of the size distribution for the grains
            sigma (float or array): Standard deviation of the size distribution

        Returns:
            array: Fourier coefficients for the size effect
        """
        # constant scalars are computed with numpy
        x1 = (self.math.log(L/m)) / (np.sqrt(2) * sigma) - 1.5 * np.sqrt(2) * sigma
        x2 = (self.math.log(L/m)) / (np.sqrt(2) * sigma) - np.sqrt(2) * sigma
        x3 = (self.math.log(L/m)) / (np.sqrt(2) * sigma)
        
        first = m**3 * self.math.exp(9 * 2 * sigma**2 / 4) * self.math.erfc(x1)/ 3
        second = m**2 * self.math.exp(2 * sigma**2) * L * self.math.erfc(x2) / 2   
        third = L**3 * self.math.erfc(x3) / 6
        norm =  m**3 * self.math.exp( 9 * 2 * sigma**2 / 4) * 2 / 3
        return (first - second + third) / norm
    
    def fourier_coefficients_dislocation(self, L, rho, Rstar, h, k, l, g, q):
        """Computes the Fourier coefficients for the dislocation profile for a given set of wave numbers, rho, Rstar, Miller indices, g and q parameters.

            Args:
                L (array): Wave numbers
                rho (float or array): dislocation density
                Rstar (float or array): cutoff radius for the dislocations
                h, k, l (int): Miller indices
                g (float): Peak location
                q (float or array): parameter characterizing the type of dislocation
            Returns:
                array: Fourier coefficients
        """
        return self.math.exp(-0.5*self.math.pi * rho * self.burgers_vector**2 * L**2 * self.wilkens_function(L, Rstar) * g**2 * self.C_h00 * self._C_hkl_without_00(h,k,l,q ))
    

    def fourier_coefficients_dislocation_reduced_rho(self, L, rhostar, Rstar, h, k, l, g, q):
        """Computes the Fourier coefficients for the dislocation profile for a given set of wave numbers, rhostar, Rstar, Miller indices, g and q parameters.
        Uses the reduced density rhostar = rho * b^2 * C_h00

            Args:
                L (array): Wave numbers
                rhostar (float or array): normalized dislocation density, rhostar = rho * b^2 * C_h00
                Rstar (float or array): cutoff radius for the dislocations
                h, k, l (int): Miller indices
                g (float): Peak location
                q (float or array): parameter characterizing the type of dislocation
            Returns:
                array: Fourier coefficients
        """
        return self.math.exp(-0.5*self.math.pi * rhostar *  L**2 * self.wilkens_function(L, Rstar) * g**2 * self._C_hkl_without_00(h, k, l, q ))
    
    def generate_convolutional_profile(self, L, m, sigma, rho_or_rhostar, Rstar, q,  g, h, k, l, planar_fault_probability = None, intrinsic_or_extrinsic = 'intrinsic'):
        """Computes the convolton of the individual profiles: dislocation, stacking fault and size effects.

        Args:
            L (array): Wave numbers
            m (float or array): median of the size distribution
            sigma (float or array): standard deviation of the size distribution
            rho_or_rhostar (float or array): dislocation density or reduced density, depending on the use_reduced_rho flag in the constructor
            Rstar (float or array): Rstar
            q (float or array): parameter characterizing the type of dislocation
            g (float): peak location
            h, k, l (int): Miller indices
            planar_fault_probability (float or array): Stacking fault probability. Only used for the fcc phase. Defaults to None.

        Returns:
            array: Convolution of the profiles
        """
        if self.use_reduced_rho: # if the reduced density is used, the dislocation profile does not depend on the contrast factor and the burgers vector
            coeff_dislocation = self.fourier_coefficients_dislocation_reduced_rho(self.math.abs(L),
                                                                   rho_or_rhostar,
                                                                     Rstar,
                                                                       h, k, l,
                                                                         g, q) + 0*1j # enforce complex type
        else:
            coeff_dislocation = self.fourier_coefficients_dislocation(self.math.abs(L),
                                                                   rho_or_rhostar,
                                                                     Rstar,
                                                                       h, k, l,
                                                                         g, q) + 0*1j 
        coeff_size = self.fourier_coefficients_size(self.math.abs(L),  m, sigma)
        # the stacking fault profile is only computed for the fcc phase. Need to loop over the subreflections
        if self.phase == 'fcc':
            if planar_fault_probability is None:
                raise ValueError('Planar fault probability must be specified for the fcc phase')
            coeff_stacking_fault = coeff_dislocation * 0 * 1j # enforce complex type
            subrefs = self.structure.subreflections['%s%s%s' %(int(h),
                                                               int(k),
                                                               int(l))]  # subreflections is a dictionary
            for subref in subrefs:
                hsub = self.math.array(subref[0]) # to make sure of proper broadcast
                ksub = self.math.array(subref[1])
                lsub = self.math.array(subref[2])
                coeff_stacking_fault += self.fourier_coefficients_planar_fault(L, 
                                                                                 planar_fault_probability,  
                                                                                 hsub, ksub, lsub, 
                                                                                 intrinsic_or_extrinsic = intrinsic_or_extrinsic) / len(subrefs)
        elif self.phase == 'bcc':
            coeff_stacking_fault = self.math.ones(coeff_dislocation.shape)
        elif self.phase == 'sc':
            coeff_stacking_fault = self.math.ones(coeff_dislocation.shape)  # for BCC and SC, this profile has no effect
        fourier_convolution =  coeff_dislocation * coeff_size * coeff_stacking_fault # multiply the three profiles elementwise to get the convolution
        N = len(L)
        dell = L[1,:] - L[0,:]
        factor = dell * N # normalization factor for the Fourier transform
        convolution_profile = self.math.abs(self.math.ifftshift( 
            self.math.ifft ( fourier_convolution , axis = 0),
              axes = 0)) * factor
        max_value = self.math.max(convolution_profile, dim = 0) # need to specify dimension, otherwise returns global max
        return convolution_profile / max_value # normalize the profile to [0,1] by default


def generate_multiple_peaks(single_peak,
                m, sigma, rho_or_rhostar, Rstar, q,
                peak_intensities,
                maximal_peakIntensity = 1, planar_fault_probability = None,
                offset = None, intrinsic_or_extrinsic = 'intrinsic'):
    """Combine multiple different peaks coming from the Peak object. 

    Args:
        single_peak (Peak): object containing the common properties of the peaks
        m (float or array): median of the size distribution
        sigma (float or array): standard deviation of the size distribution
        rho_or_rhostar (float or array): dislocation density or reduced density, depending on the use_reduced_rho flag in the constructor
        Rstar (float or array): Cutoff radius for the dislocations
        q (float or array): _description_
        peakIntensities (array): same shape as single_peak.structure.miller_indices, contains the peak intensities
        maximalPeakIntensity (int, optional): Normalize the largest peak to this value. Defaults to 1.
        planarFaultProbability (array, optional): Probability of planar faults for the fcc phase. Defaults to None.
        offset (array, optional): same shape as peak_intensities. Offset of the peak location with respect to the theoretical one. which is h^2 + k^2 + l^2. Defaults to None.
        intrinsic_or_extrinsic (string, optional): Can be 'intrinsic' or 'extrinsic' for stacking faults. Defaults to 'intrinsic'.
    Returns:
        array: spectrum 

    """
    numberofPeaks = len(peak_intensities)
    if numberofPeaks > len(single_peak.structure.miller_indices):
            print('Too many peaks were given for the structure. Max. number of peaks is %s' %len(single_peak.structure.miller_indices))
            raise NotImplementedError

    if offset is None:
        offset = single_peak.math.zeros(peak_intensities.shape) # default is no offset from theoretical value

    lengthOfFrame = 2 * single_peak.max_range_diffraction_vector

    spectrum = 1j*single_peak.math.zeros((single_peak.diffraction_vectors.shape[0], peak_intensities.shape[1])) # need this to be a complex array
    L = single_peak.math.fftfreq(single_peak.Nfourier, lengthOfFrame / single_peak.Nfourier).reshape(-1, 1) + 1e-7 # because of a singularity in the wilkens() fn, we regularize L
    for i in range(numberofPeaks):
        h = single_peak.math.array(single_peak.structure.miller_indices[i][0]) # get Miller indicesto make sure of proper broadcast
        k = single_peak.math.array(single_peak.structure.miller_indices[i][1])
        l = single_peak.math.array(single_peak.structure.miller_indices[i][2])
        sqrsum = h**2 + k**2 + l**2 + offset[i, :]
        g = sqrsum
        intensity = peak_intensities[i, :]  * maximal_peakIntensity
        singleSpectrum =  single_peak.generate_convolutional_profile(L, m, sigma, rho_or_rhostar, Rstar, q,  g, h, k, l, planar_fault_probability,
                                                                     intrinsic_or_extrinsic = intrinsic_or_extrinsic)
        indextoshift = single_peak.math.get_index_to_shift(single_peak.diffraction_vectors.reshape(-1, 1), g) # select the indices to shift, given as minima of s - g
        singleSpectrum = single_peak.math.shift_each_column(singleSpectrum, 
                                               -1*indextoshift,
                                                ) # shift the center of the peak to g. Using the custom function shift_each_column instead of np.roll 
        spectrum += intensity * singleSpectrum 

    return single_peak.math.flip_each_column(
        single_peak.math.real(
            spectrum[single_peak.diffraction_vectors >=0])) # flip the spectrum and take the real part. Discard values corresponding to negative diffraction vector lengths