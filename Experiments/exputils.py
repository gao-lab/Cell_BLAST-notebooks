import sys
import numpy as np
import numba
import scipy.stats

sys.path.insert(0, "../../Evaluation")
import utils
clean_dataset = utils.clean_dataset
na_mask = utils.na_mask


def _energy_distance_impl(x, y):  # pragma: no cover
    x_sorter = np.argsort(x)
    y_sorter = np.argsort(y)
    xy = np.concatenate((x, y))
    xy.sort()
    deltas = np.diff(xy)
    x_cdf = np.searchsorted(x[x_sorter], xy[:-1], "right") / x.size
    y_cdf = np.searchsorted(y[y_sorter], xy[:-1], "right") / y.size
    return np.sqrt(2 * np.sum(np.multiply(np.square(x_cdf - y_cdf), deltas)))


def _wasserstein_distance_impl(x, y):  # pragma: no cover
    x_sorter = np.argsort(x)
    y_sorter = np.argsort(y)
    xy = np.concatenate((x, y))
    xy.sort()
    deltas = np.diff(xy)
    x_cdf = np.searchsorted(x[x_sorter], xy[:-1], 'right') / x.size
    y_cdf = np.searchsorted(y[y_sorter], xy[:-1], 'right') / y.size
    return np.sum(np.multiply(np.abs(x_cdf - y_cdf), deltas))


@numba.extending.overload(
    scipy.stats.energy_distance, jit_options={"nogil": True, "cache": True})
def _energy_distance(x, y):  # pragma: no cover
    if x == numba.float32[::1] and y == numba.float32[::1]:
        return _energy_distance_impl


@numba.extending.overload(
    scipy.stats.wasserstein_distance, jit_options={"nogil": True, "cache": True})
def _wasserstein_distance(x, y):  # pragma: no cover
    if x == numba.float32[::1] and y == numba.float32[::1]:
        return _wasserstein_distance_impl


@numba.jit(nopython=True, nogil=True, cache=True)
def jsd_sample_fast(m1, v1, m2, v2, n_sample):
    np.random.seed(0)  # local after compile
    n, dim = m1.shape
    dl2pi = dim * np.log(2 * np.pi)
    sd1 = np.sqrt(v1)
    sd2 = np.sqrt(v2)
    jsd = np.empty(n)
    for i in range(n):
        s1 = np.empty((n_sample, dim))
        s2 = np.empty((n_sample, dim))
        for j in range(dim):
            s1[:, j] = np.random.normal(m1[i, j], sd1[i, j], size=n_sample)
            s2[:, j] = np.random.normal(m2[i, j], sd2[i, j], size=n_sample)
        lprod_v1 = np.sum(np.log(v1[i]))
        lprod_v2 = np.sum(np.log(v2[i]))
        lp1s1 = -0.5 * (
            np.sum(np.square(s1 - m1[i]) / v1[i], axis=1) +
            dl2pi + lprod_v1
        )
        lp1s2 = -0.5 * (
            np.sum(np.square(s2 - m1[i]) / v1[i], axis=1) +
            dl2pi + lprod_v1
        )
        lp2s1 = -0.5 * (
            np.sum(np.square(s1 - m2[i]) / v2[i], axis=1) +
            dl2pi + lprod_v2
        )
        lp2s2 = -0.5 * (
            np.sum(np.square(s2 - m2[i]) / v2[i], axis=1) +
            dl2pi + lprod_v2
        )
        lp12s1 = np.log(0.5 * (np.exp(lp1s1) + np.exp(lp2s1)))
        lp12s2 = np.log(0.5 * (np.exp(lp1s2) + np.exp(lp2s2)))
        kl1 = np.mean(lp1s1 - lp12s1)
        kl2 = np.mean(lp2s2 - lp12s2)
        jsd[i] = 0.5 * (kl1 + kl2)
    return jsd


@numba.jit(nopython=True, nogil=True, cache=True)
def normalized_projection_distance(
    clean1, clean2, noisy1, noisy2, normalize=True
):
    """
    ref_clean : n * latent_dim
    query_clean : n * latent_dim
    ref_noisy : n * n_posterior_samples * latent_dim
    query_noisy : n * n_posterior_samples * latent_dim
    """
    n = clean1.shape[0]
    dist = np.empty(n)
    for i in range(n):  # hit index
        x = clean1[i]  # latent_dim
        y = clean2[i]  # latent_dim
        noisy_x = noisy1[i]  # n_posterior_samples * latent_dim
        noisy_y = noisy2[i]  # n_posterior_samples * latent_dim

        projection = (x - y).reshape((-1, 1))  # latent_dim * 1
        if np.all(projection == 0):
            projection[...] = 1  # any projection is equivalent
        projection /= np.linalg.norm(projection)
        scalar_noisy_x = np.dot(noisy_x, projection).ravel()  # n_posterior_samples
        scalar_noisy_y = np.dot(noisy_y, projection).ravel()  # n_posterior_samples
        noisy_xy = np.concatenate((
            scalar_noisy_x, scalar_noisy_y
        ), axis=0)
        if normalize:
            noisy_xy1 = (noisy_xy - np.mean(scalar_noisy_x)) / np.std(scalar_noisy_x)
            noisy_xy2 = (noisy_xy - np.mean(scalar_noisy_y)) / np.std(scalar_noisy_y)
            dist[i] = scipy.stats.wasserstein_distance(
                noisy_xy1[:len(scalar_noisy_x)],
                noisy_xy1[-len(scalar_noisy_y):]
            ) + scipy.stats.wasserstein_distance(
                noisy_xy2[:len(scalar_noisy_x)],
                noisy_xy2[-len(scalar_noisy_y):]
            )
            dist[i] /= 2
        else:
            dist[i] = scipy.stats.wasserstein_distance(
                noisy_xy[:len(scalar_noisy_x)],
                noisy_xy[-len(scalar_noisy_y):]
            )
    return dist


def subsample_roc(fpr, tpr, subsample_size=1000):
    dlength = np.concatenate([
        np.zeros(1),
        np.sqrt(np.square(fpr[1:] - fpr[:-1]) + np.square(tpr[1:] - tpr[:-1]))
    ], axis=0)
    length = dlength.sum()
    step = length / subsample_size
    cumlength = dlength.cumsum()
    nstep = np.floor(cumlength / step)
    landmark = np.concatenate([np.zeros(1).astype(np.bool_), nstep[1:] == nstep[:-1]], axis=0)
    return fpr[~landmark], tpr[~landmark]