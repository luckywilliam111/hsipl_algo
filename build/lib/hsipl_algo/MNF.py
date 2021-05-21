# -*- coding: utf-8 -*-
"""
Created on Sun May 16 20:45:45 2021

@author: WEN
"""

import numpy as np
import collections

class LinearTransform:
    def __init__(self, A, **kwargs):

        self._pre = kwargs.get('pre', None)
        self._post = kwargs.get('post', None)
        A = np.array(A, copy=True)
        if A.ndim == 0:
            self._A = A
            (self.dim_out, self.dim_in) = (None, None)
        else:
            if len(A.shape) == 1:
                self._A = A.reshape(((1,) + A.shape))
            else:
                self._A = A
            (self.dim_out, self.dim_in) = self._A.shape
        self.dtype = kwargs.get('dtype', self._A.dtype)

    def __call__(self, X):
        if not isinstance(X, np.ndarray):
            if hasattr(X, 'transform') and isinstance(X.transform, collections.Callable):
                return X.transform(self)
            else:
                raise TypeError('Unable to apply transform to object.')

        shape = X.shape
        if len(shape) == 3:
            X = X.reshape((-1, shape[-1]))
            if self._pre is not None:
                X = X + self._pre
            Y = np.dot(self._A, X.T).T
            if self._post is not None:
                Y += self._post
            return Y.reshape((shape[:2] + (-1,))).squeeze().astype(self.dtype)
        else:
            if self._pre is not None:
                X = X + self._pre
            Y = np.dot(self._A, X.T).T
            if self._post is not None:
                Y += self._post
            return Y.astype(self.dtype)

    def chain(self, transform):
        if isinstance(transform, np.ndarray):
            transform = LinearTransform(transform)
        if self.dim_in is not None and transform.dim_out is not None \
                and self.dim_in != transform.dim_out:
            raise Exception('Input/Output dimensions of chained transforms'
                            'do not match.')

        if transform._pre is not None:
            pre = np.array(transform._pre)
        else:
            pre = None
        post = None
        if transform._post is not None:
            post = np.array(transform._post)
            if self._pre is not None:
                post += self._pre
        elif self._pre is not None:
            post = np.array(self._pre)
        if post is not None:
            post = self._A.dot(post)
        if self._post:
            post += self._post
        if post is not None:
            post = np.array(post)
        A = np.dot(self._A, transform._A)
        return LinearTransform(A, pre=pre, post=post)

def matrix_sqrt(X=None, symmetric=False, inverse=False, eigs=None):
    if eigs is not None:
        (vals, V) = eigs
    else:
        (vals, V) = np.linalg.eig(X)
    
    if inverse is False:
        SRV = np.diag(np.sqrt(vals))
    else:
        SRV = np.diag(1. / np.sqrt(vals))
    if symmetric:
        return V.dot(SRV).dot(V.T)
    else:
        return V.dot(SRV).dot(np.linalg.inv(V))

class Iterator:
    def __init__(self):
        pass

    def __iter__(self):
        raise NotImplementedError('Must override __iter__ in child class.')

    def get_num_elements(self):
        raise NotImplementedError(
            'Must override get_num_elements in child class.')

    def get_num_bands(self):
        raise NotImplementedError(
            'Must override get_num_bands in child class.')

class GaussianStats(object):
    def __init__(self, mean=None, cov=None, nsamples=None, inv_cov=None):
        self.cov = cov
        self._inv_cov = inv_cov
        self.mean = mean
        self.nsamples = nsamples

    @property
    def cov(self):
        return self._cov

    @cov.setter
    def cov(self, C):
        self.reset_derived_stats()
        self._cov = C

    @property
    def inv_cov(self):
        if self._inv_cov is None:
            self._inv_cov = np.linalg.inv(self._cov)
        return self._inv_cov

    def reset_derived_stats(self):
        self._cov = self._inv_cov = None
        self._sqrt_cov = self._sqrt_inv_cov = self._pcs = None
        self._log_det_cov = None

    @property
    def sqrt_cov(self):
        if self._sqrt_cov is None:
            pcs = self.principal_components
            self._sqrt_cov = matrix_sqrt(eigs=(pcs.eigenvalues,
                                               pcs.eigenvectors),
                                         symmetric=True)
        return self._sqrt_cov

    @property
    def sqrt_inv_cov(self):
        if self._sqrt_inv_cov is None:
            pcs = self.principal_components
            self._sqrt_inv_cov = matrix_sqrt(eigs=(pcs.eigenvalues, pcs.eigenvectors), symmetric=True, inverse=True)
        return self._sqrt_inv_cov

    @property
    def principal_components(self):
        if self._pcs is None:
            (evals, evecs) = np.linalg.eigh(self._cov)
            self._pcs = PrincipalComponents(evals, evecs, self)
        return self._pcs

    @property
    def log_det_cov(self):
        if self._log_det_cov is None:
            evals = self.principal_components.eigenvalues
            self._log_det_cov = np.sum(np.log([v for v in evals if v > 0]))
        return self._log_det_cov

    def transform(self, xform):
        if not isinstance(xform, LinearTransform):
            raise TypeError('Expected a LinearTransform object.')
        m = xform(self.mean)
        C = xform._A.dot(self.cov).dot(xform._A.T)
        return GaussianStats(mean=m, cov=C, nsamples=self.nsamples)

    def get_whitening_transform(self):
        C_1 = np.linalg.inv(self.cov)
        return LinearTransform(matrix_sqrt(C_1, True), pre=-self.mean)

class PrincipalComponents:
    def __init__(self, vals, vecs, stats):
        self.eigenvalues = vals
        self.eigenvectors = vecs
        self.stats = stats
        self.transform = LinearTransform(self.eigenvectors.T, pre=-self.mean)

    @property
    def mean(self):
        return self.stats.mean

    @property
    def cov(self):
        return self.stats.cov

    def reduce(self, N=0, **kwargs):
        num = kwargs.get('num', None)
        eigs = kwargs.get('eigs', None)
        fraction = kwargs.get('fraction', None)
        if num is not None:
            return PrincipalComponents(self.eigenvalues[:num], self.eigenvectors[:, :num], self.stats)
        elif eigs is not None:
            vals = self.eigenvalues[eigs]
            vecs = self.eigenvectors[:, eigs]
            return PrincipalComponents(vals, vecs, self.stats)
        elif fraction is not None:
            if not 0 < fraction <= 1:
                raise Exception('fraction must be in range (0,1].')
            N = len(self.eigenvalues)
            cumsum = np.cumsum(self.eigenvalues)
            sum = cumsum[-1]
            
            for i in range(N):
                if (cumsum[i] / sum) >= fraction:
                    break
            if i == (N - 1):
                return self

            vals = self.eigenvalues[:i + 1]
            vecs = self.eigenvectors[:, :i + 1]
            return PrincipalComponents(vals, vecs, self.stats)
        else:
            raise Exception('Must specify one of the following keywords:'
                            '`num`, `eigs`, `fraction`.')

    def denoise(self, X, **kwargs):
        f = self.get_denoising_transform(**kwargs)
        return f(X)

    def get_denoising_transform(self, **kwargs):
        V = self.reduce(self, **kwargs).eigenvectors
        f = LinearTransform(V.dot(V.T), pre=-self.mean, post=self.mean)
        return f


def principal_components(image):
    if isinstance(image, GaussianStats):
        stats = image
    else:
        stats = calc_stats(image)

    (L, V) = np.linalg.eig(stats.cov)
    
    if not np.alltrue(np.diff(L) <= 0):
        ii = list(reversed(np.argsort(L)))
        L = L[ii]
        V = V[:, ii]

    return PrincipalComponents(L, V, stats)

def has_nan(X):
    return bool(np.isnan(np.min(X)))

class NaNValueError(ValueError):
    pass

class ImageIterator(Iterator):
    def __init__(self, im):
        self.image = im
        self.numElements = im.shape[0] * im.shape[1]

    def get_num_elements(self):
        return self.numElements

    def get_num_bands(self):
        return self.image.shape[2]

    def __iter__(self):
        (M, N) = self.image.shape[:2]
        
        for i in range(M):
            self.row = i
            for j in range(N):
                self.col = j
                yield self.image[i, j]

class ImageMaskIterator(Iterator):
    def __init__(self, image, mask, index=None):
        if mask.shape != image.shape[:len(mask.shape)]:
            raise ValueError('Mask shape does not match image.')
        self.image = image
        self.index = index
        
        if index:
            self.mask = np.equal(mask, index)
        else:
            self.mask = np.not_equal(mask, 0)
        self.n_elements = sum(self.mask.ravel())

    def get_num_elements(self):
        return self.n_elements

    def get_num_bands(self):
        return self.image.shape[2]

    def __iter__(self):
        coords = np.argwhere(self.mask)
        for (i, j) in coords:
            (self.row, self.col) = (i, j)
            yield self.image[i, j].astype(self.image.dtype).squeeze()

def iterator(image, mask=None, index=None):
    if isinstance(image, Iterator):
        return image
    elif mask is not None:
        return ImageMaskIterator(image, mask, index)
    else:
        return ImageIterator(image)

def mean_cov(image, mask=None, index=None):
    if isinstance(image, np.ndarray):
        X = image.astype(np.float64)
        if X.ndim == 3:
            X = image.reshape(-1, image.shape[-1]).T
        if mask is not None:
            mask = mask.ravel()
            if index is not None:
                ii = np.argwhere(mask == index)
            else:
                ii = np.argwhere(mask != 0)
            X = np.take(X, ii.squeeze(), axis=1)
        m = np.average(X, axis=1)
        C = np.cov(X)
        return (m, C, X.shape[1])

    if not isinstance(image, Iterator):
        it = iterator(image, mask, index)
    else:
        it = image
    
    B = it.get_num_bands()

    sumX = np.zeros((B,), 'd')
    sumX2 = np.zeros((B, B), 'd')
    count = 0
    
    for x in it:
        count += 1
        sumX += x
        x = x.astype(np.float64)[:, np.newaxis]
        sumX2 += x.dot(x.T)
    mean = (sumX / count)
    sumX = sumX[:, np.newaxis]
    cov = (sumX2 - sumX.dot(sumX.T) / count) / (count - 1)
    return (mean, cov, count)

#==============================================================================
#==================================== MNF =====================================
#==============================================================================

class MNFResult(object):
    def __init__(self, signal, noise, napc):
        self.signal = signal
        self.noise = noise
        self.napc = napc

    def _num_from_kwargs(self, **kwargs):
        for key in kwargs:
            if key not in ('num', 'snr'):
                raise Exception('Keyword not recognized.')
        num = kwargs.get('num', None)
        snr = kwargs.get('snr', None)
        if num == snr == None:
            raise Exception('Must specify either `num` or `snr` keyword.')
        if None not in (num, snr):
            raise Exception('Can not specify both `num` and `snr` keywords.')
        if snr is not None:
            num = self.num_with_snr(snr)
        return num

    def denoise(self, X, **kwargs):
        f = self.get_denoising_transform(**kwargs)
        return f(X)

    def get_denoising_transform(self, **kwargs):
        N = self._num_from_kwargs(**kwargs)
        V = self.napc.eigenvectors
        Vr = np.array(V)
        Vr[:, N:] = 0.
        f = LinearTransform(self.noise.sqrt_cov.dot(Vr).dot(V.T) \
			    .dot(self.noise.sqrt_inv_cov),
                            pre=-self.signal.mean,
                            post=self.signal.mean)
        return f

    def reduce(self, X, **kwargs):
        f = self.get_reduction_transform(**kwargs)
        return f(X)

    def get_reduction_transform(self, **kwargs):
        N = self._num_from_kwargs(**kwargs)
        V = self.napc.eigenvectors
        f = LinearTransform(V[:, :N].T.dot(self.noise.sqrt_inv_cov),
                            pre=-self.signal.mean)
        return f

    def num_with_snr(self, snr):
        return np.sum(self.napc.eigenvalues >= (snr + 1))

def mnf(signal, noise):
    C = noise.sqrt_inv_cov.dot(signal.cov).dot(noise.sqrt_inv_cov)
    (L, V) = np.linalg.eig(C)
    
    if not np.alltrue(np.diff(L) <= 0):
        ii = list(reversed(np.argsort(L)))
        L = L[ii]
        V = V[:, ii]
    wstats = GaussianStats(mean=np.zeros_like(L), cov=C)
    napc = PrincipalComponents(L, V, wstats)
    return MNFResult(signal, noise, napc)

def calc_stats(image, mask=None, index=None, allow_nan=False):
    (mean, cov, N) = mean_cov(image, mask, index)
    if has_nan(mean) and not allow_nan:
        raise NaNValueError('NaN values present in data.')
    return GaussianStats(mean=mean, cov=cov, nsamples=N)

def noise_from_diffs(X, direction='lowerright'):
    if direction.lower() not in ['lowerright', 'lowerleft', 'right', 'lower']:
        raise ValueError('Invalid `direction` value.')
    if direction == 'lowerright':
        deltas = X[:-1, :-1, :] - X[1:, 1:, :]
    elif direction == 'lowerleft':
        deltas = X[:-1, 1:, :] - X[1:, :-1, :]
    elif direction == 'right':
        deltas = X[:, :-1, :] - X[:, 1:, :]
    else:
        deltas = X[:-1, :, :] - X[1:, :, :]

    stats = calc_stats(deltas)
    stats.cov /= 2.0
    return stats

#==============================================================================
#==============================================================================
#==============================================================================