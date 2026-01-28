import numpy as np


def whiten(X):
    """
    Given a matrix X, returns the whitened version, where the steps include:

    1.) Mean shifting: to make the later eigen decomposition unbiased from the origin
    2.) Compute covariance of data
    3.) Compute the eigenvectors of the covariance matrix
    4.) Compute the whitening matrix
    5.) Apply the whitening matrix to the data
    """
    #1
    X_mean = np.expand_dims(np.mean(X, axis=1), axis=1)
    X = X - X_mean
    #X = mean_shift(X)
    #2
    cov = X @ X.T / X.shape[1] #calculating covariance over time, so we must divide by number of timepoints
    #3
    D, E = np.linalg.eigh(cov)
    idx = D.argsort()[::-1]
    D, E = D[idx], E[:, idx]
    #4
    D_inv_sqrt = np.diag(1 / np.sqrt(D))
    W = D_inv_sqrt @ E.T #whitening matrix
    #5
    Z = W @ X

    return Z, E, D, X_mean

def g(x):
    """
    Derivative of log(cohs(x))
    """
    return np.tanh(x)

def g_prime(x):
    """
    Derivative of tanh(x)
    """
    return 1 - g(x) **2

def symmetric_orthogonalization(W):
    """
    Makes vectors orthogonal while minimizing strucutral distortion
    """
    U,_,Vt = np.linalg.svd(W, full_matrices=False)
    return U @ Vt

def ICA(Z, max_iter=1000, n_components=3, tol=1e-10):
    """
    Z is the whitened data, of shape k x M
    """

    n_features, n_samples = Z.shape

    #assert n_components <= n_features

    W = np.random.normal(size=(n_components, n_features))
    W = symmetric_orthogonalization(W)
    for i in range(max_iter):
        Wold = W
        Y = W @ Z
        G = g(Y)
        G_prime = g_prime(Y)
        t1 = G @ Z.T / n_samples
        t2 = np.mean(G_prime, axis=1, keepdims=True) * W
        W = t1 - t2
        W = symmetric_orthogonalization(W)
        sim = np.abs(np.diag(W @ Wold.T))
        if np.max(np.abs(sim - 1)) < tol:
            print(f"Converged after {i+1} iters")
            break
    return W

def invert_transform(Z, w, D, E, X_mean):
    """
    Z is the whitened data
    w is the vector in question
    D are the eigenvalues from whitening
    E are the eigenvectors
    X_mean is the original data mean
    """
    Q = E @ np.diag(np.sqrt(D)) #inverse whitening
    mixing_vector = Q @ w
    source = w @ Z
    X_single_component = np.outer(mixing_vector, source) + X_mean
    return X_single_component, source

def run_ICA(X, n_components=3, max_iter=1000, returns=None):
    """
    Runs ICA on a given data matrix

    Inputs:
    X: N x M, where N is the number of observations and M is the number of data points for each observation
    n_components: number of components to identify
    max_iter: maximum number of iterations for ICA
    returns: dictionary of what data to return
    
    Outputs:
    A: mixing matrix
    S: independent components
    E: eigenvectors for whitening
    D: eigenvalues for whitening
    Z: whitened data
    W: vectors that identify components
    """
    if returns is None:
        returns = {x: True for x in ['W', 'Z', 'S', 'A', 'E', 'D']}

    Z, E, D, X_mean = whiten(X)
    W = ICA(Z, n_components=n_components, max_iter=max_iter)
    S = W @ Z
    A = E @ np.diag(np.sqrt(D)) @ W.T
    return_dict = {}
    values_map = {'W': W, 'Z': Z, 'S': S, 'A': A, 'E': E, 'D': D}
    for key, val in values_map.items():
        if returns.get(key, False):
            return_dict[key] = val
    return return_dict
    
