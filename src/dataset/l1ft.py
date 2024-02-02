import pandas as pd
import numpy as np
from itertools import chain
from cvxopt import solvers
from cvxopt import matrix, spmatrix
from matplotlib import pyplot as plt
from statsmodels.robust import mad

solvers.options['show_progress'] = 0


def get_second_derivative_matrix(n):
    """
    :param n: The size of the time series

    :return: A matrix D such that if x.size == (n,1), D * x is the second derivate of x
    """
    m = n - 2
    D = spmatrix(list(chain(*[[1, -2, 1]] * m)),
                 list(chain(*[[i] * 3 for i in range(m)])),
                 list(chain(*[[i, i + 1, i + 2] for i in range(m)])))
    return D


def _l1tf(corr, delta):
    """
        minimize    (1/2) * ||x-corr||_2^2 + delta * sum(y)
        subject to  -y <= D*x <= y

    Variables x (n), y (n-2).

    :param x:
    :return:
    """

    n = corr.size[0]
    m = n - 2

    D = get_second_derivative_matrix(n)

    P = D * D.T
    q = -D * corr

    G = spmatrix([], [], [], (2*m, m))
    G[:m, :m] = spmatrix(1.0, range(m), range(m))
    G[m:, :m] = -spmatrix(1.0, range(m), range(m))

    h = matrix(delta, (2*m, 1), tc='d')

    res = solvers.qp(P, q, G, h)

    return corr - D.T * res['x']


def l1tf(corr, delta):
    """
    :param corr: Corrupted signal, should be a numpy array / pandas Series
    :param delta: Strength of regularization

    :return: The filtered series
    """

    m = float(corr.min())
    M = float(corr.max())
    denom = M - m
    # if denom == 0, corr is constant
    t = (corr-m) / (1 if denom == 0 else denom)

    if isinstance(corr, np.ndarray):
        values = matrix(t)
    elif isinstance(corr, pd.Series):
        values = matrix(t.values[:])
    else:
        raise ValueError("Wrong type for corr")

    values = _l1tf(values, delta)
    values = values * (M - m) + m

    if isinstance(corr, np.ndarray):
        values = np.asarray(values).squeeze()
    elif isinstance(corr, pd.Series):
        values = pd.Series(values, index=corr.index, name=corr.name)

    return values


def remove_outliers(t, delta, mad_factor=3):
    """
    :param t: an instance of pd.Series
    :param delta: parameter for l1tf function
    """
    filtered_t = l1tf(t, delta)

    diff = t.values - np.asarray(filtered_t).squeeze()
    t = t.copy()
    t[np.abs(diff - np.median(diff)) > mad_factor * mad(diff)] = np.nan

    t = t.fillna(method='ffill').fillna(method='bfill')
    return t


def strip_na(s):
    """
    :param s: an instance of pd.Series

    Removes the NaN from the extremes
    """
    m = s.min()
    lmask = s.fillna(method='ffill').fillna(m-1) == m-1
    rmask = s.fillna(method='bfill').fillna(m-1) == m-1
    mask = np.logical_or(lmask, rmask)
    return s[np.logical_not(mask)]


def df_l1tf(df, delta=3, remove_outliers=False, mad_factor=3):
    """
    Applies the l1tf function to the whole dataframe optionally removing outliers

    :param df: A pandas Dataframe
    :param delta: The delta parameter of the l1tf function
    :param remove_outliers: Whether outliers should be removed
    :param mad_factor: Strength of the outlier detection technique
    """
    l1tf_d = {}
    if remove_outliers: wo_outliers_d = {}
    ks = df.keys()

    for i, k in enumerate(ks):
        if i % 50 == 0: print(i, 'of', len(ks))
        t = strip_na(df[k])

        if remove_outliers:
            t = remove_outliers(t, delta, mad_factor)
            wo_outliers_d[k] = t
        s = l1tf(t, delta)
        l1tf_d[k] = s

    if remove_outliers:
        return pd.DataFrame(l1tf_d), pd.DataFrame(wo_outliers_d)
    else:
        return pd.DataFrame(l1tf_d)


if __name__ == '__main__':
    s = np.hstack([
        np.arange(1, 100) + np.random.normal(0, 10, 99),
        np.arange(100, 50, step=-1) + np.random.normal(0, 10, 50),
        np.arange(300, 400) + np.random.normal(0, 10, 100),
    ])
    plt.plot(s)
    tf = l1tf(s, 0.9)
    plt.plot(tf)
    plt.show()