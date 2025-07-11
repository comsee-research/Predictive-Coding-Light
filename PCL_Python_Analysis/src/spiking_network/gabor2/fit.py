import theano
import theano.tensor as T
from scipy.optimize import minimize
from scipy.ndimage.filters import gaussian_filter as gf
from numpy import fft

import numpy as np


def setup_graph():
    """Setup the theano graph for all possible operations."""
    n_x = T.lscalar('n_x')
    n_y = T.lscalar('n_y')
    pos_x = T.arange(n_x).dimshuffle(0, 'x', 'x')
    pos_y = T.arange(n_y).dimshuffle('x', 0, 'x')

    params = T.dvector('params')
    s_params = split_params(params)
    x, y, theta, phi, lkx, lvx, lvy = s_params

    xp = x.dimshuffle('x', 'x', 0)
    yp = y.dimshuffle('x', 'x', 0)
    thetap = theta.dimshuffle('x', 'x', 0)
    phip = phi.dimshuffle('x', 'x', 0)
    lkxp = lkx.dimshuffle('x', 'x', 0)
    lkxp = 2. * np.pi / (2. + T.exp(lkxp))
    lvxp = lvx.dimshuffle('x', 'x', 0)
    lvyp = lvy.dimshuffle('x', 'x', 0)

    x_prime = T.cos(theta)*(pos_x-x) -T.sin(theta)*(pos_y-y)
    y_prime = T.sin(theta)*(pos_x-x) +T.cos(theta)*(pos_y-y)

    envelope = T.exp(-x_prime**2/T.exp(lvxp)/2.-y_prime**2/T.exp(lvyp)/2.)
    phase = T.sin(lkxp * x_prime+phip)
    gabor = envelope * phase

    gabor_norm = T.sqrt((gabor**2).sum(axis=(0, 1), keepdims=True))
    envelope_norm = T.sqrt((envelope**2).sum(axis=(0, 1), keepdims=True))
    phase_norm = T.sqrt((phase**2).sum(axis=(0, 1), keepdims=True))

    gabor = gabor/gabor_norm
    envelope = envelope/envelope_norm
    phase = phase/phase_norm

    return params, s_params, n_x, n_y, gabor, envelope, phase

def fit_lvx_lvy_function(data):
    params, s_params, n_x, n_y, gabor, envelope, phase = setup_graph()
    x, y, theta, phi, lkx, lvx, lvy = s_params

    se = ((data-gabor)**2).sum(axis=(0, 1))
    mse = se.mean().astype('float64')
    grad = T.grad(mse, params, consider_constant=(x, y, theta, phi, lkx))
    return params, mse, se, grad, gabor, n_x, n_y

def fit_theta_phi_lkx_function(data):
    params, s_params, n_x, n_y, gabor, envelope, phase = setup_graph()
    x, y, theta, phi, lkx, lvx, lvy = s_params

    se = ((data-gabor)**2).sum(axis=(0, 1))
    mse = se.mean().astype('float64')
    grad = T.grad(mse, params, consider_constant=(x, y, lvx, lvy))
    return params, mse, se, grad, gabor, n_x, n_y

def fit_theta_phi_function(data):
    params, s_params, n_x, n_y, gabor, envelope, phase = setup_graph()
    x, y, theta, phi, lkx, lvx, lvy = s_params

    se = ((data-phase)**2).sum(axis=(0, 1))
    mse = se.mean().astype('float64')
    grad = T.grad(mse, params, consider_constant=[lkx])
    return params, mse, se, grad, phase, n_x, n_y

def fit_only_envelope_function(data):
    params, s_params, n_x, n_y, gabor, envelope, phase = setup_graph()

    se = ((data-envelope)**2).sum(axis=(0, 1))
    mse = se.mean().astype('float64')
    grad = T.grad(mse, params)
    return params, mse, se, grad, envelope, n_x, n_y

def fit_x_y_function(data):
    params, s_params, n_x, n_y, gabor, envelope, phase = setup_graph()
    x, y, theta, phi, lkx, lvx, lvy = s_params

    se = ((data-envelope)**2).sum(axis=(0, 1))
    mse = se.mean().astype('float64')
    grad = T.grad(mse, params, consider_constant=[theta, lvx, lvy])
    return params, mse, se, grad, gabor, n_x, n_y

def fit_phi_x_y_function(data):
    params, s_params, n_x, n_y, gabor, envelope, phase = setup_graph()
    x, y, theta, phi, lkx, lvx, lvy = s_params

    se = ((data-gabor)**2).sum(axis=(0, 1))
    mse = se.mean().astype('float64')
    grad = T.grad(mse, params, consider_constant=[theta, lkx, lvx, lvy])
    return params, mse, se, grad, gabor, n_x, n_y

def fit_envelope_function(data):
    params, s_params, n_x, n_y, gabor, envelope, phase = setup_graph()
    x, y, theta, phi, lkx, lvx, lvy = s_params

    se = ((data-gabor)**2).sum(axis=(0, 1))
    mse = se.mean().astype('float64')
    grad = T.grad(mse, params, consider_constant=[theta, lkx])
    return params, mse, se, grad, gabor, n_x, n_y

def fit_all_function(data):
    params, s_params, n_x, n_y, gabor, envelope, phase = setup_graph()

    se = ((data-gabor)**2).sum(axis=(0, 1))
    mse = se.mean().astype('float64')
    grad = T.grad(mse, params)
    return params, mse, se, grad, gabor, n_x, n_y


def combine_params(x, y, theta, phi, lkx, lvx, lvy):
    """Turns individual parameter vectors into a parameter array."""
    if isinstance(x, theano.tensor.TensorVariable):
        rval = T.concatenate([x, y, theta, phi, lkx, lvx, lvy])
    else:
        rval = np.concatenate([x, y, theta, phi, lkx, lvx, lvy])
    return rval

def split_params(params):
    """Splits a parameter vector for a batch of gabors into individual parameter
    vectors."""
    n_samples = params.shape[0]//7
    x = params[:n_samples].astype('float32')
    y = params[n_samples:2*n_samples].astype('float32')
    theta = params[2*n_samples:3*n_samples].astype('float32')
    phi = params[3*n_samples:4*n_samples].astype('float32')
    lkx = params[4*n_samples:5*n_samples].astype('float32')
    lvx = params[5*n_samples:6*n_samples].astype('float32')
    lvy = params[6*n_samples:].astype('float32')
    return x, y, theta, phi, lkx, lvx, lvy

def standardize_params(*params):
    """Convert parameters from internal representation to standard Gabor
    parameters.

    Parameters
    ----------
    x, y, theta, phi, lkx, lvx, lvy
        Either a combines vector or split parameters.

    Returns
    -------
    x : float
       Center of the Gabor in the x direction in pixels.
    y : float
       Center of the Gabor in the y direction in pixels.
    theta : float
       Rotation of the Gabor in the plane.
    phi : float
       Phase of the Gabor.
    kx : float
       Wavevector of Gabor (2*pi/lambda).
    vx : float
        Variance of the Gabor along the oscilation direction.
    vy : float
        Variance of the Gabor perpendictular to the oscilation direction.
    """
    combine = False
    if len(params) == 1:
        x, y, theta, phi, lkx, lvx, lvy = split_params(*params)
        combine = True
    else:
        x, y, theta, phi, lkx, lvx, lvy = params
    if isinstance(x, theano.tensor.TensorVariable):
        kx = 2.*np.pi / (2.*np.sqrt(2)+T.exp(lkx))
        rval = x, y, theta, phi, kx, T.exp(lvx), T.exp(lvy)
    else:
        kx = 2.*np.pi / (2.*np.sqrt(2)+np.exp(lkx))
        rval = x, y, theta, phi, kx, np.exp(lvx), np.exp(lvy)
    if combine:
        rval = combine_params(*rval)
    return rval


class GaborFit(object):
    """Fit Gabor parameters to patches and visualize Gabors."""
    def __init__(self):
        self.data = theano.shared(np.empty((1,1,1), dtype='float32'))

        (params, mse, se, grad, gabor,
         n_x_s, n_y_s)  = fit_x_y_function(self.data)
        self._fit_x_y = theano.function([params], [mse, grad],
                                        givens={n_x_s: self.data.shape[0],
                                                n_y_s: self.data.shape[1]})
        self._fit_x_y_se = theano.function([params], se,
                                           givens={n_x_s: self.data.shape[0],
                                                   n_y_s: self.data.shape[1]})

        (params, mse, se, grad, gabor,
         n_x_s, n_y_s)  = fit_phi_x_y_function(self.data)
        self._fit_phi_x_y = theano.function([params], [mse, grad],
                                            givens={n_x_s: self.data.shape[0],
                                                    n_y_s: self.data.shape[1]})
        self._fit_phi_x_y_se = theano.function([params], se,
                                               givens={n_x_s: self.data.shape[0],
                                                       n_y_s: self.data.shape[1]})

        (params, mse, se, grad, gabor,
         n_x_s, n_y_s)  = fit_theta_phi_lkx_function(self.data)
        self._fit_theta_phi_lkx = theano.function([params], [mse, grad],
                                                  givens={n_x_s: self.data.shape[0],
                                                          n_y_s: self.data.shape[1]})

        self._fit_theta_phi_lkx_se = theano.function([params], se,
                                                     givens={n_x_s: self.data.shape[0],
                                                             n_y_s: self.data.shape[1]})

        (params, mse, se, grad, gabor,
         n_x_s, n_y_s)  = fit_all_function(self.data)
        self._fit_all = theano.function([params], [mse, grad],
                                        givens={n_x_s: self.data.shape[0],
                                                n_y_s: self.data.shape[1]})
        self._fit_all_se = theano.function([params], se,
                                           givens={n_x_s: self.data.shape[0],
                                                   n_y_s: self.data.shape[1]})

        (params, mse, se, grad, gabor,
         n_x_s, n_y_s)  = fit_lvx_lvy_function(self.data)
        self._fit_lvx_lvy = theano.function([params], [mse, grad],
                                            givens={n_x_s: self.data.shape[0],
                                                    n_y_s: self.data.shape[1]})
        self._fit_lvx_lvy_se = theano.function([params], se,
                                               givens={n_x_s: self.data.shape[0],
                                                       n_y_s: self.data.shape[1]})

        params_s, s_params, n_x_s, n_y_s, gabor, envelope, phase = setup_graph()
        self._make_gabor = theano.function([params_s, n_x_s, n_y_s], gabor)
        self._make_phase = theano.function([params_s, n_x_s, n_y_s], phase)
        self._make_envelope = theano.function([params_s, n_x_s, n_y_s], envelope)

    def fit(self, X, var_init=.05):
        """Given image patches, find best-fit Gabor parameters.

        Parameters
        ----------
        X : ndarray (n_x, n_y, n_batch)
            Image patches for fitting.
        var_init : float
            Ballpark variance initialization scaled by dim**2.

        Returns
        -------
        x : list
            List of all parameter setting during fitting.
        best_params : ndarray
            Internal parameter vector for best parameters.
        best_se : ndarray
            Squared-error for best parameters settings for each element of the
            batch.
        """
        # Calculate different versions of the data
        n_x, n_y, n_samples = X.shape
        init = np.zeros(7*n_samples)

        X_norm = np.sqrt((X**2).sum(axis=(0, 1), keepdims=True))
        X = X/X_norm

        fao = np.array([gf(abs(xi), 2, mode='constant', cval=0.)
                        for xi in X.transpose(2, 0, 1)]).transpose(1, 2, 0).astype('float32')

        fao_norm = np.sqrt((fao**2).sum(axis=(0, 1), keepdims=True))
        fao = fao/fao_norm

        aps = abs(fft.fft2(X, axes=(0, 1)))
        aps_norm = np.sqrt((aps**2).sum(axis=(0, 1), keepdims=True))
        aps = aps/aps_norm
        freqs = fft.fftfreq(n_x)[:, np.newaxis] + 1j*fft.fftfreq(n_y)[np.newaxis, :]

        thetas = np.linspace(0., np.pi, 8)
        kx_min = 2.*np.pi/np.sqrt(n_x**2+n_y**2)
        kx_max = 2.*np.pi/2./np.sqrt(2.)
        kxs = np.linspace(kx_min, kx_max, 20, endpoint=True)
        lkxs = np.log(2.*np.pi/kxs + 2.*np.sqrt(2))

        def choose_best(best_se, best_params, se, params):
            compare = se < best_se
            best_params = best_params.reshape(7, -1)
            params = params.reshape(7, -1)
            best_se[compare] = se[compare]
            best_params[:, compare] = params[:, compare]
            return best_se, best_params.ravel()

        best_se = np.inf*np.ones(n_samples)
        best_params = np.zeros(7*n_samples)

        x = []
        for vi in [var_init/2., var_init, 2.*var_init]:
            init = np.zeros(7*n_samples)
            init[:n_samples] = n_x/2.
            init[n_samples:2*n_samples] = n_y/2.
            init[4*n_samples:5*n_samples] = lkxs[0]
            init[5*n_samples:6*n_samples] = np.log(vi*(n_x)**2)
            init[6*n_samples:7*n_samples] = np.log(vi*(n_y)**2)

            # Fit envelope mean
            func = self._fit_x_y
            self.data.set_value(fao.astype('float32'))
            res = minimize(func, init, method='L-BFGS-B', jac=True)
            x.append(res.x)
            params = res.x

            x.append(best_params)

            self.data.set_value(X.astype('float32'))

            func = self._fit_theta_phi_lkx
            func_se = self._fit_theta_phi_lkx_se
            for theta in thetas:
                for lkx in lkxs:
                    init = params.copy()
                    init[2*n_samples:3*n_samples] = theta
                    init[4*n_samples:5*n_samples] = lkx
                    res = minimize(func, init, method='L-BFGS-B', jac=True)
                    params = res.x
                    se = func_se(params)
                    best_se, best_params = choose_best(best_se, best_params, se, params)
                    x.append(best_params)
                #print theta, k, se.mean(), best_se.mean()

        # Fit envelope widths
        func = self._fit_lvx_lvy
        res = minimize(func, best_params, method='L-BFGS-B', jac=True)
        params = res.x
        se = self._fit_lvx_lvy_se(params)
        best_se, best_params = choose_best(best_se, best_params, se, params)

        x.append(best_params)

        # Fit envelope center and phase
        func = self._fit_phi_x_y
        res = minimize(func, best_params, method='L-BFGS-B', jac=True)
        params = res.x
        se = self._fit_phi_x_y_se(params)
        best_se, best_params = choose_best(best_se, best_params, se, params)

        x.append(best_params)

        # Fit envelope center and phase
        func = self._fit_all
        res = minimize(func, best_params, method='L-BFGS-B', jac=True)
        params = res.x
        se = self._fit_all_se(params)
        best_se, best_params = choose_best(best_se, best_params, se, params)

        x.append(best_params)

        return x, split_params(best_params), best_se

    def make_gabor(self, params, n_x, n_y):
        return self._make_gabor(params, n_x, n_y)

    def make_phase(self, params, n_x, n_y):
        return self._make_phase(params, n_x, n_y)

    def make_envelope(self, params, n_x, n_y):
        return self._make_envelope(params, n_x, n_y)