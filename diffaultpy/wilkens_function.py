from diffaultpy.math_backend import MathBackend
from diffaultpy.utils import exact_integral_for_wilkens
import numpy as np
from scipy.interpolate import interp1d 


class WilkensFunction():

    def __init__(self, backend, method, device, dtype):
        self.method = method
        self.math = MathBackend(backend, device, dtype)
        self.integrator_for_wilkens = None

    def initialize_with_method(self):
        """Initializes the Wilkens function with the specified method for approximating the integral of arcsin(x) / x. This can be an exact computation, which is scipy's numerical integrator, a polynomial approximation or an interpolation. Only the polynomial approximation is implemented for the torch backend."""
        if self.method == 'exact':
            # set the integrator for the Wilkens function to the exact integral
            self.integrator_for_wilkens = exact_integral_for_wilkens
            if self.math.backend == 'torch':
                raise NotImplementedError('Exact computation of Wilkens function is not implemented for the torch backend')
            self.method = 'wilkens'
        elif self.method == 'polynomial':
            xx_fit = np.linspace(1e-8, 1, 1000) # there is a singularity at x=0
            integral = exact_integral_for_wilkens(xx_fit)
            polynomial_fit = np.polyfit(xx_fit, integral, deg = 5) # Fit a 5th order polynomial to the integral  
            coefficients = polynomial_fit[::-1] # reverse the order due to polyfit convention
            coefficients = self.math.array(coefficients.copy())
            self.integrator_for_wilkens = lambda x : self.math.power(x.reshape(-1,1),
                                                            self.math.arange(0,6)) @ coefficients 
        elif self.method == 'interpolation':
            if self.math.backend == 'torch':
                raise NotImplementedError('Interpolation of the Wilkens function is not implemented for the torch backend')
            xx_fit = np.linspace(1e-8, 1, 1000) # there is a singularity at x=0
            integral = exact_integral_for_wilkens(xx_fit)
            integralInterpolated = interp1d(np.linspace(0, 1, 1000), integral, fill_value = 'extrapolate') 
            self.integrator_for_wilkens = integralInterpolated
        return
    
    def __call__(self, L, Rstar):
        """Computes the Wilkens function for a given set of wave numbers and Rstar parameter, which is needed for the dislocation profile.
        The Wilkens function is a function of the ratio L/Rstar. The variable is denoted as eta = 0.5 * e^-1/4 * L / Rstar.It is a piecewise-defined function, therefore the computation is split into two parts, depending on the value of eta.
        Depending on self.approximation_wilkens, the integral of arcsin(x) / x can be computed exactly, approximated by a polynomial or interpolated. 

        Args:
            L (array): Wave numbers
            Rstar (double): 

        Returns:
            array: value of the Wilkens function
        """
        eta = 0.5 * np.exp(-0.25) * L / Rstar # scaled L / Rstar
        # compute the integral of arcsin x / x from 0 to eta for eta < 1
        eta_smaller_than_1 = self.math.where(eta < 1, eta, 1)
        integral = self.integrator_for_wilkens(eta_smaller_than_1).reshape(eta_smaller_than_1.shape)

        row1 = -self.math.log(eta_smaller_than_1) + (7. / 4. - np.log(2)) 
        row2 = 512. / (90 * self.math.pi) * 1. / eta_smaller_than_1 + (2. / self.math.pi) * (1 - 1./( 4 * eta_smaller_than_1**2 )) * integral
        row31 = -(1. / self.math.pi) * (769 / (180.) * 1. / eta_smaller_than_1 + 41. * eta_smaller_than_1 / 90 + 2 * eta_smaller_than_1**3 / 90.) * self.math.sqrt(1 - eta_smaller_than_1**2)
        row32 = -(1. / self.math.pi) * (11. / 12 * (1. / eta_smaller_than_1**2) + 7. / 2. + eta_smaller_than_1**2 / 3.) * self.math.arcsin(eta_smaller_than_1) + eta_smaller_than_1**2 / 6
        wilk_value_when_smaller = row1 + row2 + row31 + row32

        eta_bigger_than_1 = self.math.where(eta >= 1, eta, 1)
        wilk_value_when_bigger =  512. / (90 * self.math.pi) * (1. / eta_bigger_than_1) - (11. / 24 + 0.25 * self.math.log(2 * eta_bigger_than_1)) * (1. / eta_bigger_than_1**2)
        return self.math.where(eta < 1, wilk_value_when_smaller, wilk_value_when_bigger)