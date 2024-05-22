from cmwp_profiles.peak_shapes import Peak, generate_multiple_peaks
import numpy as np
import torch
import matplotlib.pyplot as plt


def test_torch_differential():
    peak_torch = Peak(14, 2048, 0.1, 0.255, 0.36, stacking_or_twin_fault='twin', backend='torch')
    lengthOfFrame = 2 * peak_torch.max_range_diffraction_vector
    L = peak_torch.math.fftfreq(peak_torch.Nfourier, lengthOfFrame / peak_torch.Nfourier).reshape(-1, 1) + 1e-7
    h = peak_torch.math.array(1)
    k = peak_torch.math.array(1)
    l = peak_torch.math.array(1)
    g = peak_torch.math.array([5])
    q = peak_torch.math.array([1.8,1.8, 1.8], requires_grad=True)
    rho = peak_torch.math.array([0.1, 0.1, 0.1], requires_grad=True)
    Rstar = peak_torch.math.array([8., 8., 8.], requires_grad=True)
    beta = peak_torch.math.array([0.02, 0.02, 0.02], requires_grad=True)
    m = peak_torch.math.array([50, 50, 50], requires_grad=True)
    sigma = peak_torch.math.array([1e-3, 1e-3, 1e-3])
    intensities = torch.tensor(np.random.rand(5,3), requires_grad=True)
    offsets = torch.tensor(np.random.rand(5,3) * 1e-5, requires_grad=True)
    multipeaks_torch = generate_multiple_peaks(peak_torch, m, sigma, beta, rho, Rstar, q, intensities, maximal_peakIntensity = 1, offset = offsets)
    value = torch.sum(multipeaks_torch**2)
    value.backward()
    assert np.allclose(np.isnan(q.grad), 0)
    assert np.allclose(q.grad.shape, q.detach().numpy().shape)


def test_torch_mse():
    peak_torch = Peak(14, 2048, 0.1, 0.255, 0.36, stacking_or_twin_fault='twin', backend='torch')
    lengthOfFrame = 2 * peak_torch.max_range_diffraction_vector
    L = peak_torch.math.fftfreq(peak_torch.Nfourier, lengthOfFrame / peak_torch.Nfourier).reshape(-1, 1) + 1e-7
    h = peak_torch.math.array(1)
    k = peak_torch.math.array(1)
    l = peak_torch.math.array(1)
    g = peak_torch.math.array([5])
    q = peak_torch.math.array([1.8,1.8, 1.8], requires_grad=True)
    rho = peak_torch.math.array([0.1, 0.1, 0.1], requires_grad=True)
    Rstar = peak_torch.math.array([8., 8., 8.], requires_grad=True)
    beta = peak_torch.math.array([0.02, 0.02, 0.02], requires_grad=True)
    m = peak_torch.math.array([50, 50, 50], requires_grad=True)
    sigma = peak_torch.math.array([1e-3, 1e-3, 1e-3])
    intensities = torch.tensor(np.random.rand(5,3), requires_grad=True)
    offsets = torch.tensor(np.random.rand(5,3) * 1e-5, requires_grad=True)
    multipeaks_torch = generate_multiple_peaks(peak_torch, m, sigma, beta, rho, Rstar, q, intensities, maximal_peakIntensity = 1, offset = offsets)
    loss = torch.nn.MSELoss()
    output = loss(multipeaks_torch, multipeaks_torch* 0)
    lsq_value = torch.mean(multipeaks_torch**2)
    assert torch.allclose(lsq_value, output)
                                                             
if __name__ == "__main__":

    test_torch_differential()