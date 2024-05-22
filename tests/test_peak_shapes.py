from diffaultpy.utils import shift_each_column_numpy, shift_each_column_torch
from diffaultpy.peak_shapes import Peak, generate_multiple_peaks
import numpy as np
import torch
import matplotlib.pyplot as plt


def test_peak_init():
    peak = Peak(14, 1024, 0.1, 0.255, 0.36)
    assert np.allclose(peak.diffraction_vectors, np.linspace(-14, 14, 1024))
    assert np.sum(peak.positive_diffraction_vectors<0)  == 0
    assert peak.planar_fault_type == 'stacking'
    return 

def test_wilkens_approx():
    peak_exact = Peak(14, 1024, 0.1, 0.255, 0.36, approximation_wilkens='exact')
    peak_interpolation = Peak(14, 1024, 0.1, 0.255, 0.36, approximation_wilkens='interpolation')
    peak_poly = Peak(14, 1024, 0.1, 0.255, 0.36, approximation_wilkens='polynomial')
    peak_poly_torch = Peak(14, 1024, 0.1, 0.255, 0.36, backend = 'torch', approximation_wilkens='polynomial')

    test_x_numpy = np.linspace(1e-3, 1, 50)
    assert np.allclose(peak_exact.wilkens_function.integrator_for_wilkens(test_x_numpy), peak_interpolation.wilkens_function.integrator_for_wilkens(test_x_numpy), atol=1e-5)
    error_numpy = peak_exact.wilkens_function.integrator_for_wilkens(test_x_numpy) - peak_poly.wilkens_function.integrator_for_wilkens(test_x_numpy)
    test_x_torch = peak_poly_torch.math.array(test_x_numpy)
    error_torch = peak_exact.wilkens_function.integrator_for_wilkens(test_x_numpy) - peak_poly_torch.wilkens_function.integrator_for_wilkens(test_x_torch).numpy()
    assert np.max(np.abs(error_numpy)) < 1e-2 # sub 1% error
    assert np.max(np.abs(error_torch)) < 1e-2  # sub 1% error
    assert np.allclose(peak_poly.wilkens_function.integrator_for_wilkens(test_x_numpy), peak_poly_torch.wilkens_function.integrator_for_wilkens(test_x_torch).numpy(), atol=1e-5) # numpy and torch should be the same
    return

def test_fwhm():
    peak = Peak(14, 1024, 0.1, 0.255, 0.36, stacking_or_twin_fault='twin')
    fwhm = peak._full_width_half_max( 0.36, 1, 1, 1, 0.01)  # divisible by 3,  returns minimial value
    assert np.allclose(fwhm, 0.01) 
    fwhm = peak._full_width_half_max( 0.36, 1, 0, 1, 0.01) 
    assert np.allclose(fwhm, 0.0263210760932852) 
    peak = Peak(14, 1024, 0.1, 0.255, 0.36, stacking_or_twin_fault='stacking')
    fwhm = peak._full_width_half_max( 0.36, 1, 0, 1, 0.01) 
    assert np.allclose(fwhm, 0.0394800984344216) 
    return

def test_delta():
    peak = Peak(14, 1024, 0.1, 0.255, 0.36, stacking_or_twin_fault='twin')
    delta_int = peak._delta_hkl(0.36, 1, 0, 1, 0.01)
    peak_torch = Peak(14, 1024, 0.1, 0.255, 0.36, approximation_wilkens='polynomial', backend='torch', dtype = torch.float32, stacking_or_twin_fault='twin')
    delta_int = peak._delta_hkl(0.36, 1, 0, 1, 0.01)
    delta_int_torch = peak_torch._delta_hkl(peak_torch.math.array(0.36), peak_torch.math.array(1), peak_torch.math.array(0), peak_torch.math.array(1), peak_torch.math.array(0.01))
    assert np.allclose(delta_int, 0.)
    assert np.allclose(delta_int_torch, 0.)
    return


def test_delta_stacking():
    peak = Peak(14, 1024, 0.1, 0.255, 0.36, stacking_or_twin_fault='stacking')
    delta_int = peak._delta_hkl(0.36, 1, 0, 1, 0.01)
    peak_torch = Peak(14, 1024, 0.1, 0.255, 0.36, approximation_wilkens='polynomial', backend='torch', dtype = torch.float32, stacking_or_twin_fault='stacking')
    delta_int = peak._delta_hkl(0.36, 1, 0, 1, 0.01)
    delta_int_torch = peak_torch._delta_hkl(peak_torch.math.array(0.36), peak_torch.math.array(1), peak_torch.math.array(0), peak_torch.math.array(1), peak_torch.math.array(0.01))
    assert np.allclose(delta_int, -0.0115126275106)
    assert np.allclose(delta_int_torch, -0.0115126275106)
    return

def test_shift_each_column():
    array_np = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).T
    shifts_np = np.array([1, -1, 0])  
    shifted_np = shift_each_column_numpy(array_np, shifts_np)
    #print("Shifted NumPy Array:\n", shifted_np)
    array_torch = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).T
    shifts_torch = torch.tensor([1, -1, 0]) 
    shifted_torch = shift_each_column_torch(array_torch, shifts_torch)
    assert np.allclose(shifted_np, shifted_torch.numpy())
    assert np.allclose(shifted_np, np.array([[3, 1, 2], [5, 6, 4], [7, 8, 9]]).T)
    #print("Shifted PyTorch Tensor:\n", shifted_torch)

    return
### test the fourier coefficients individually

def test_fourier_coefficients_stacking_fault():
    peak = Peak(14, 1024, 0.1, 0.255, 0.36, stacking_or_twin_fault='twin')
    peak_torch = Peak(14, 1024, 0.1, 0.255, 0.36, stacking_or_twin_fault='twin', backend='torch')
    lengthOfFrame = 2 * peak.max_range_diffraction_vector
    L = peak.math.fftfreq(peak.Nfourier, lengthOfFrame / peak.Nfourier).reshape(-1, 1) + 1e-14
    h = peak.math.array(1)
    k = peak.math.array(0)
    l = peak.math.array(1)
    beta = peak.math.array([0.02, 0.02, 0.02])

    profile_stacking = peak.fourier_coefficients_planar_fault(L, beta, h, k, l)
    lengthOfFrame = 2 * peak_torch.max_range_diffraction_vector
    L = peak_torch.math.fftfreq(peak_torch.Nfourier, lengthOfFrame / peak_torch.Nfourier).reshape(-1,1 ) + 1e-14
    h = peak_torch.math.array(1)
    k = peak_torch.math.array(0)
    l = peak_torch.math.array(1)
    beta = peak_torch.math.array([0.02, 0.02, 0.02])
    profile_stacking_torch = peak_torch.fourier_coefficients_planar_fault(L, beta, h, k, l)
    assert np.allclose(profile_stacking, profile_stacking_torch.numpy(), atol=1e-5)
    return

def test_fourier_coefficients_size():
    peak = Peak(14, 2048, 0.1, 0.255, 0.36, stacking_or_twin_fault='twin')
    peak_torch = Peak(14, 2048, 0.1, 0.255, 0.36, stacking_or_twin_fault='twin', backend='torch')
    lengthOfFrame = 2 * peak.max_range_diffraction_vector
    L = peak.math.fftfreq(peak.Nfourier, lengthOfFrame / peak.Nfourier).reshape(-1, 1) + 1e-16
    m = peak.math.array([0.02, 0.02, 0.02])
    sigma = peak.math.array([1e-3, 1e-3, 1e-3])
    profile_size = peak.fourier_coefficients_size(peak.math.abs(L), m, sigma)
    #print(profile_size)
    #print(profile_size.shape) 
    lengthOfFrame = 2 * peak_torch.max_range_diffraction_vector
    L = peak_torch.math.fftfreq(peak_torch.Nfourier, lengthOfFrame / peak_torch.Nfourier).reshape(-1,1 ) + 1e-16
    m = peak_torch.math.array([0.02, 0.02, 0.02])
    sigma = peak_torch.math.array([1e-3, 1e-3, 1e-3])
    profile_size_torch = peak_torch.fourier_coefficients_size(peak_torch.math.abs(L), m, sigma)
    assert np.allclose(profile_size, profile_size_torch.numpy(), atol=1e-5)

def test_fourier_coefficients_dislocation():
    peak = Peak(14, 2048, 0.1, 0.255, 0.36, stacking_or_twin_fault='twin')
    peak_torch = Peak(14, 2048, 0.1, 0.255, 0.36, stacking_or_twin_fault='twin', backend='torch')
    lengthOfFrame = 2 * peak.max_range_diffraction_vector
    L = peak.math.fftfreq(peak.Nfourier, lengthOfFrame / peak.Nfourier).reshape(-1, 1) + 1e-7
    h = peak.math.array(1)
    k = peak.math.array(0)
    l = peak.math.array(1)
    g = peak.math.array([1.])
    q = peak.math.array([1.8])
    rho = peak.math.array([0.1, 0.1, 0.1])
    Rstar = peak.math.array([8., 8., 8.])
    profile_dislocation = peak.fourier_coefficients_dislocation(peak.math.abs(L), rho, Rstar, h, k, l, g, q)

    lengthOfFrame = 2 * peak_torch.max_range_diffraction_vector
    L = peak_torch.math.fftfreq(peak_torch.Nfourier, lengthOfFrame / peak_torch.Nfourier).reshape(-1, 1) + 1e-7
    h = peak_torch.math.array(1)
    k = peak_torch.math.array(0)
    l = peak_torch.math.array(1)
    g = peak_torch.math.array([1])
    q = peak_torch.math.array([1.8])
    rho = peak_torch.math.array([0.1, 0.1, 0.1])
    Rstar = peak_torch.math.array([8., 8., 8.])
    profile_dislocation_torch = peak_torch.fourier_coefficients_dislocation(peak_torch.math.abs(L), rho, Rstar, h, k, l, g, q)
#    plt.plot(profile_dislocation[:,0])
#    plt.plot(profile_dislocation_torch[:,0].numpy())
#    plt.show()
    assert np.allclose(profile_dislocation, profile_dislocation_torch.numpy(), atol=1e-5)


def test_get_index_to_shift():
    s = np.linspace(0, 14, 100)
    s0 = np.array([2, 3, 4])
    peak = Peak(14, 2048, 0.1, 0.255, 0.36, stacking_or_twin_fault='twin')

    idx = peak.math.get_index_to_shift(s.reshape(-1,1), s0)
    print(idx)

def test_generate_convolutional_profile():
    peak = Peak(14, 2048, 0.1, 0.255, 0.36, stacking_or_twin_fault='twin')
    peak_torch = Peak(14, 2048, 0.1, 0.255, 0.36, stacking_or_twin_fault='twin', backend='torch')
    lengthOfFrame = 2 * peak.max_range_diffraction_vector
    L = peak.math.fftfreq(peak.Nfourier, lengthOfFrame / peak.Nfourier).reshape(-1, 1) + 1e-7
    h = peak.math.array([1])
    k = peak.math.array([1])
    l = peak.math.array([1])
    
    g = peak.math.array([5.3, 5.3, 5.3])
    q = peak.math.array([1.8])
    rho = peak.math.array([0.1, 0.1, 0.1])
    Rstar = peak.math.array([8., 8., 8.])
    beta = peak.math.array([0.02, 0.02, 0.02])
    m = peak.math.array([50, 50, 50])
    sigma = peak.math.array([1e-3, 1e-3, 1e-3])
    profile = peak.generate_convolutional_profile(L, m, sigma, rho, Rstar, q, g, h, k, l, planar_fault_probability = beta)
    lengthOfFrame = 2 * peak_torch.max_range_diffraction_vector
    L = peak_torch.math.fftfreq(peak_torch.Nfourier, lengthOfFrame / peak_torch.Nfourier).reshape(-1, 1) + 1e-7
    h = peak_torch.math.array([1])
    k = peak_torch.math.array([1])
    l = peak_torch.math.array([1])
    g = peak_torch.math.array([5.3, 5.3, 5.3])
    q = peak_torch.math.array([1.8])
    rho = peak_torch.math.array([0.1, 0.1, 0.1])
    Rstar = peak_torch.math.array([8., 8., 8.])
    beta = peak_torch.math.array([0.02, 0.02, 0.02])
    m = peak_torch.math.array([50,50, 50])
    sigma = peak_torch.math.array([1e-3, 1e-3, 1e-3])
    profile_torch = peak_torch.generate_convolutional_profile(L, m, sigma, rho, Rstar, q, g, h, k, l, planar_fault_probability = beta)

    assert np.allclose(profile, profile_torch.numpy(), atol=1e-5)
    return 

def test_generate_multiple_peaks():
    peak = Peak(14, 2048, 0.1, 0.255, 0.36, stacking_or_twin_fault='twin')
    peak_torch = Peak(14, 2048, 0.1, 0.255, 0.36, stacking_or_twin_fault='twin', backend='torch')
    lengthOfFrame = 2 * peak.max_range_diffraction_vector
    L = peak.math.fftfreq(peak.Nfourier, lengthOfFrame / peak.Nfourier).reshape(-1, 1) + 1e-7
    h = peak.math.array(1)
    k = peak.math.array(1)
    l = peak.math.array(1)
    g = peak.math.array([5])
    q = peak.math.array([1.8])
    rho = peak.math.array([0.1, 0.1, 0.1])
    Rstar = peak.math.array([8., 8., 8.])
    beta = peak.math.array([0.02, 0.02, 0.02])
    m = peak.math.array([50, 50, 50])
    sigma = peak.math.array([1e-3, 1e-3, 1e-3])
    intensities = np.random.rand(5,3)
    offsets = np.random.rand(5,3) * 1e-5
    multipeaks = generate_multiple_peaks(peak, m, sigma, rho, Rstar, q, intensities, maximal_peakIntensity = 1, offset = offsets, planar_fault_probability = beta)  
    
    lengthOfFrame = 2 * peak_torch.max_range_diffraction_vector
    L = peak_torch.math.fftfreq(peak_torch.Nfourier, lengthOfFrame / peak_torch.Nfourier).reshape(-1, 1) + 1e-7
    h = peak_torch.math.array([1])
    k = peak_torch.math.array([1])
    l = peak_torch.math.array([1])
    g = peak_torch.math.array([5])
    q = peak_torch.math.array([1.8])
    rho = peak_torch.math.array([0.1, 0.1, 0.1])
    Rstar = peak_torch.math.array([8., 8., 8.])
    beta = peak_torch.math.array([0.02, 0.02, 0.02])
    m = peak_torch.math.array([50,50, 50])
    sigma = peak_torch.math.array([1e-3, 1e-3, 1e-3])
    intensities = torch.tensor(intensities)
    offsets = torch.tensor(offsets)
    multipeaks_torch = generate_multiple_peaks(peak_torch, m, sigma, rho, Rstar, q, intensities, maximal_peakIntensity = 1, offset = offsets, planar_fault_probability = beta)
    #for i in range(3):
    #    plt.plot(peak.positive_diffraction_vectors, multipeaks[:,i])
    #plt.show()
    #for i in range(3):
    #    plt.plot(peak_torch.positive_diffraction_vectors, multipeaks_torch[:,i].numpy())
    #plt.show()
    assert np.allclose(multipeaks, multipeaks_torch.numpy(), atol=1e-5)
    return 

                               
#if __name__ == "__main__":
    #test_generate_multiple_peaks()
    #test_generate_convolutional_profile()
    #test_delta()
    # test_peak_init()
    # test_wilkens_approx()
    # test_fwhm()
    # test_delta()
    # test_fourier_coefficients_stacking_fault()
    # test_fourier_coefficients_size()
    # test_fourier_coefficients_dislocation()
    # test_generate_convolutional_profile()
    # test_get_index_to_shift()
    # test_shift_each_column()
    # test_generate_multiple_peaks()
