
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from tqdm import tqdm  # Importer tqdm pour afficher la barre de progression

NVAR = 2                        # generation de données en 2D seulement



def generatesources(Nsources, Nsamples, choice):
    """ Generates sources with independent components (for ICA/BSS)

    Returned sources are normalized.

    Parameters
    ----------
    Nsources : int, number of sources
    Nsamples : int, number of samples
    choice : selected type of sources: 'bpsk', 'rand'

    Returns
    -------
    sources : ndarray Nsour x Nsamples
              realization of the sources
    """
    if choice == 'bpsk':
        sources = np.sign(np.random.randn(Nsources, Nsamples))
    elif choice == 'rand':
        sources = np.random.rand(Nsources, Nsamples) - 0.5

    # renormalize (unit variance)
    if Nsamples == 0:
        sources = np.empty((Nsources, 0))
    else:
        renormfactors = 1/np.sqrt((sources**2).sum(axis=1)/Nsamples)
        sources = sources*np.repeat(renormfactors[:, np.newaxis], Nsamples,
                                    axis=1)
    return sources






def elimamb(x: np.array, y: np.array):
    """ Eliminate permutation/sign ambiguities in ICA and compte MSE

    Parameters
    ----------
    x : array Nx x Nsamples
    y : array Ny x Nsamples with Ny = Nx

    Returns
    -------
    mse : array Nx
    P : array Nx x Nx
    """
    assert (x.shape == y.shape)
    Nsources = x.shape[0]
    Nsamples = x.shape[1]
    r = x@y.T
    P = np.zeros(r.shape)
    for _ in range(Nsources):
        indmax = np.unravel_index(abs(r).argmax(), r.shape)
        P[indmax] = np.sign(r[indmax])
        r[indmax[0], :] = 0
        r[:, indmax[1]] = 0
    newx = x
    newy = P@y
    mse = ((newx-newy)**2).sum(axis=1)/Nsamples

    return mse, P



def genlatentbin(nsamples, eta, exact=False):
    """
    Generate latent (hidden) binary 1D process.

    Each entry is Bernouilli with P(False) = eta, P(True)=1-eta
    TODO: extend e.g. Markov dependence

    Parameters
    ----------
    nsamples : int
    eta : float 0<eta<1
    exact : (optional). It False (default), each sample is randomly driven.
            If exact=True, exactly int(eta*nsamples) are set to False.

    Returns
    -------
    r : ndarray, shape (nsamples, ) with bool entries
    """
    if exact is False:
        r = np.random.rand(nsamples) > eta
    elif exact is True:
        nsamples0 = int(nsamples*eta)
        nsamples1 = nsamples-nsamples0
        r = np.r_[np.repeat(False, nsamples0), np.repeat(True, nsamples1)]
        np.random.default_rng().shuffle(r)  # in-place shuffling
    return r


def mog(nsamples, mu=0.5, sigma=None):
    """
    Generates samples in R from a specific mixture of Gaussians in ref. [1]

    1/2*N(mu,sigma^2) + 1/2*N(-mu,sigma^2)]    with:  0<mu<1
    Default value for sigma is sigma = sqrt(1 - mu^2) to ensure unit variance

    Parameters
    ----------
    nsamples : int
    mu : float
         0<mu<1. Default: mu = 0.5 (value used in [1])
    sigma : float
            Default: sigma = None and is calculated to ensure unit variance

    Returns
    -------
    a : ndarray, shape (nsamples, )

    References
    ----------
    [1] Castella, M., Rafi, S., Comon, P., & Pieczynski, W. (2013). Separation
        of instantaneous mixtures of a particular set of dependent sources
        using classical ICA methods. EURASIP J. Adv. Signal Process., (62), .
    """
    if sigma is None:
        sigma = np.sqrt(1-mu**2)
    a = mu*np.sign(np.random.randn(nsamples)) + sigma*np.random.randn(nsamples)
    return a


def dependsourex1(nsamples, mu=0.5):
    """
    Generates dependent sources according to example 1 in ref. [1]

    Sources are given by s1 = a  s2=epsilon*a, where a obtained by mog.

    Parameters
    ----------
    nsamples : int
    mu : parameter value for mog (mixture of Gaussian)

    Returns
    -------
    s : ndarray, shape (2, nsamples)

    References
    ----------
    [1] Castella, M., Rafi, S., Comon, P., & Pieczynski, W. (2013). Separation
        of instantaneous mixtures of a particular set of dependent sources
        using classical ICA methods. EURASIP J. Adv. Signal Process., (62), .

    TODO:
    -----
    instead of mu, entry of function should be function generating samples
    """
    a = mog(nsamples, mu)
    epsilon = np.sign(np.random.randn(nsamples))
    s = np.c_[a, epsilon*a].T
    return s


def dependsourex2(nsamples, lambd):
    """
    Generates dependent sources according to example 2 in ref. [1]

    Parameters
    ----------
    nsamples : int

    Returns
    -------
    s : ndarray, shape (2, nsamples)

    References
    ----------
    [1] Castella, M., Rafi, S., Comon, P., & Pieczynski, W. (2013). Separation
        of instantaneous mixtures of a particular set of dependent sources
        using classical ICA methods. EURASIP J. Adv. Signal Process., (62), .
    """
    sigma = np.sqrt(2*(1-1/lambd**2))
    u1 = sigma*np.random.randn(nsamples)
    u2 = np.random.default_rng().laplace(0, 1/lambd, nsamples)
    U = np.c_[u1, u2].T
    rot = 1/np.sqrt(2)*np.array([[1, -1],
                                 [1, 1]])
    # ### ci-dessous, cas exact article [1]. Rq: det(rot)<0. Ne change rien ici
    # rot = 1/np.sqrt(2)*np.array([[1, 1],
    #                              [1, -1]])
    s = rot@U
    return s


def dependsourex3(nsamples, beta, thirdsourcewidth=0, thirdsource=True):
    """
    Generate two or three dependent sources according to:

    s2 = 1/beta^2*s1^3
    with s1 in [-beta, beta] and hence s2 in [-beta, beta]
    s3 uniform on [-thirdsourcewidth,+thirdsourcewidth] (if non zero)
    s3 not given by default with thirdsourcewidth=0

    Parameters
    ----------
    nsamples : int
    beta : float (should be positive)
    thirdsourcewidth : float (default = 0)
    thirdsource : bool (default = True)

    Returns
    -------
    s : ndarray, shape (nsour, nsamples) with nsour = 3 if thirdsource=True
                                              nsour = 2 if thirdsource=False
    """
    s1 = 2*np.random.rand(1, nsamples) - 1
    s1 = s1*beta
    s2 = 1/(beta**2)*s1**3
    s = np.r_[s1, s2]
    if thirdsource is True:
        s3 = thirdsourcewidth*(2*np.random.rand(1, nsamples) - 1)
        s = np.r_[s, s3]
        s = s[[0, 2, 1], :]
    return s





def tfuni4(e):
    """Replicates tfuni4.m : from CoM2 algorithm for ICA by P. Comon

    Orthogonal real direct transform for separating 2 sources in presence of
    noise. Sources are assumed zero mean.

    Parameters
    ----------
    e : (2,T)-array

    Returns
    -------
    S : (2,T)-array shape
    A : (2,2)-array shape

    """
    T = max(e.shape)
    # % moments d'ordre 2
    g11 = sum(e[0, :]*e[0, :])/T        # cv vers 1
    g22 = sum(e[1, :]*e[1, :])/T        # cv vers 1
    g12 = sum(e[0, :]*e[1, :])/T        # cv vers 0
    # % moments d'ordre 4
    e2 = e**2
    g1111 = sum(e2[0, :]*e2[0, :])/T
    g1112 = sum(e2[0, :]*e[0, :]*e[1, :])/T
    g1122 = sum(e2[0, :]*e2[1, :])/T
    g1222 = sum(e2[1, :]*e[1, :]*e[0, :])/T
    g2222 = sum(e2[1, :]*e2[1, :])/T
    # % cumulants croises d'ordre 4
    q1111 = g1111-3*g11*g11
    q1112 = g1112-3*g11*g12
    q1122 = g1122-g11*g22-2*g12*g12
    q1222 = g1222-3*g22*g12
    q2222 = g2222-3*g22*g22
    # % racine de Pw(x): si t est la tangente de l'angle, x=t-1/t.
    u = q1111+q2222-6*q1122
    v = q1222-q1112
    z = q1111*q1111+q2222*q2222

    c4 = q1111*q1112-q2222*q1222
    c3 = z-4*(q1112*q1112+q1222*q1222)-3*q1122*(q1111+q2222)
    c2 = 3*v*u
    c1 = 3*z-2*q1111*q2222-32*q1112*q1222-36*q1122*q1122
    c0 = -4*(u*v+4*c4)

    Pw = np.array([c4, c3, c2, c1, c0])
    R = np.roots(Pw)
    float_epsilon = np.finfo(float).eps
    xx = R[abs(R.imag) < float_epsilon].real
    # % maximum du contraste en x
    a0 = q1111
    a1 = 4*q1112
    a2 = 6*q1122
    a3 = 4*q1222
    a4 = q2222
    b4 = a0*a0+a4*a4
    b3 = 2*(a3*a4-a0*a1)
    b2 = 4*a0*a0+4*a4*a4+a1*a1+a3*a3+2*a0*a2+2*a2*a4
    b1 = 2*(-3*a0*a1+3*a3*a4+a1*a4+a2*a3-a0*a3-a1*a2)
    b0 = 2*(a0*a0+a1*a1+a2*a2+a3*a3+a4*a4+2*a0*a2+2*a0*a4+2*a1*a3+2*a2*a4)

    Pk = [b4, b3, b2, b1, b0]   # numerateur du contraste
    Wk = np.polyval(Pk, xx)
    Vk = np.polyval([1, 0, 8, 0, 16], xx)
    Wk = Wk/Vk
    Xsol = xx[Wk == max(Wk)]      # [Wmax,j]=max(Wk); Xsol=xx(j);
    # % maximum du contraste en theta
    t = np.roots(np.array([1, -Xsol[0], -1]))
    mask = np.all([-1 < t, t <= 1], axis=0)
    t = t[mask]
    # % test et conditionnement
    if abs(t) < 1/T:
        A = np.eye(2)
        # fprintf('pas de rotation plane pour cette paire\n');
    else:
        A = np.empty([2, 2])
        A[0, 0] = 1/np.sqrt(1+t*t)
        A[1, 1] = A[0, 0]
        A[0, 1] = t*A[0, 0]
        A[1, 0] = -A[0, 1]
    # % filtrage de la sortie
    S = A@e

    return S, A
def comica(Y: np.array):
    """ ICA algorithm Com2 by Pierre Comon (Adapted, original code in Matlab).

    REFERENCE: P.Comon, "Independent Component Analysis, a new concept?",
    Signal Processing, Elsevier, vol.36, no 3, April 1994, 287-314.

    Parameters
    ----------
    Y : array, shape(Ny, Nsamples) where Ny << Nsamples with observations

    Returns
    -------
    F : array, shape(Ny, Ny): estimated mixing matrix in model Y=FS
    """
    if Y.shape[0] > Y.shape[1]:
        Y = Y.transpose()
    N, T = Y.shape              # Y est maintenant NxT avec N<T.
    # % STEPS 1 & 2: whitening and projection (PCA)
    U, s, V = np.linalg.svd(Y.T, full_matrices=False)
    # ATTENTION: difference svd wrt Matlab: s vector, V<- Vh
    tol = 0                     # original prog: tol=max(size(S))*norm(S)*eps;
    mask = s > tol
    U = U[:, mask]
    V = (V.T)[:, mask]
    S = np.diag(s[mask])
    r = U.shape[1]
    Z = U.T*np.sqrt(T)
    L = V@S.T/np.sqrt(T)
    F = L
    # % STEPS 3 & 4 & 5: Unitary transform
    S = Z
    if N == 2:
        K = 1
    else:
        K = int(1 + np.round(np.sqrt(N)))  # max number of sweeps
    Rot = np.eye(r)
    for k in range(K):
        Q = np.eye(r)
        for i in range(r-1):
            for j in range(i+1, r):
                S1ij = np.vstack((S[i, :], S[j, :]))
                Sij, qij = tfuni4(S1ij)  # %%%%%% processing a pair
                S[i, :] = Sij[0, :]
                S[j, :] = Sij[1, :]
                Qij = np.eye(r)
                Qij[i, i] = qij[0, 0]
                Qij[i, j] = qij[0, 1]
                Qij[j, i] = qij[1, 0]
                Qij[j, j] = qij[1, 1]
                Q = Qij@Q
        Rot = Rot@Q.T
    F = F@Rot
    return F



def forward_backward(T, Netats, Pxr, pihat, Phat):
    """Perform the forward-backward algorithm to compute the state probabilities and psi.

    Parameters
    ----------
    T : int
        Number of samples.
    Netats : int
        Number of states.
    Pxr : ndarray
        State observation probabilities.
    pihat : ndarray
        Initial state probabilities.
    Phat : ndarray
        State transition matrix.

    Returns
    -------
    alpha : ndarray
        Forward probabilities.
    beta : ndarray
        Backward probabilities.
    gamma : ndarray
        Smoothed state probabilities.
    psi : ndarray
        Pairwise state probabilities.
    """
    alpha = np.zeros((Netats, T))
    beta = np.zeros((Netats, T))
    alpha[:, 0] = pihat * Pxr[:, 0]
    alpha[:, 0] /= np.sum(alpha[:, 0])  

    for t in range(1, T):
        alpha[:, t] = np.dot(alpha[:, t-1], Phat.T) * Pxr[:, t]
        alpha[:, t] /= np.sum(alpha[:, t])  

    beta[:, T-1] = 1
    for t in range(T-2, -1, -1):
        beta[:, t] = np.dot(Phat, beta[:, t+1] * Pxr[:, t+1])
        coeffrenorm = np.dot(alpha[:, t], np.dot(Phat, Pxr[:, t+1])) 
        beta[:, t] /= coeffrenorm

    gamma = alpha * beta
    sum_gamma = np.sum(gamma, axis=0)
    sum_gamma[sum_gamma == 0] = 1  # Avoid division by zero
    gamma /= sum_gamma  # Normalize gamma
    psi = np.zeros((Netats, Netats, T-1))
    for t in range(T-1):
        psi[:, :, t] = np.outer(alpha[:, t], beta[:, t+1] * Pxr[:, t+1]) * Phat
        coeffrenorm = np.sum(psi[:, :, t])
        if coeffrenorm == 0:  # Avoid division by zero
            coeffrenorm = 1
        psi[:, :, t] /= coeffrenorm

    return alpha, beta, gamma, psi

def generate_markov_process(T, P):
    """Generate a sequence of states using a Markov process.

    Parameters
    ----------
    T : int
        Number of time steps.
    P : ndarray
        State transition probability matrix.

    Returns
    -------
    r : ndarray
        Sequence of states generated by the Markov process.
    """
    Netats = P.shape[0]
    r = np.zeros(T, dtype=int)
    r[0] = np.random.choice(Netats)  # Random initial state
    for t in range(1, T):
        r[t] = np.random.choice(Netats, p=P[r[t-1]])
    return r


def monte_carlo_simulation( Netats, n_realizations, T_values=[1000,10000],lambda_ICE=15):
    
    """Performs a Monte Carlo simulation to evaluate the performance of ICA/ICE algorithm.

    Parameters
    ----------
    Netats : int
        Number of states.
    n_realizations : int
        Number of realizations for the simulation.
    T_values : list of int, optional
        List of different sample sizes to evaluate. Default is [1000, 10000].
    lambda_ICE : float, optional
        Regularization parameter for ICE. Default is 15.

    Returns
    -------
    None
    """
    fig_mse, axes_mse = plt.subplots(1, len(T_values), figsize=(14, 6))
    fig_segrate, axes_segrate = plt.subplots(1, len(T_values), figsize=(14, 6))
    
    if len(T_values) == 1:
        axes_mse = [axes_mse]
        axes_segrate = [axes_segrate]
    
    for i, T in enumerate(T_values):
        mse_ice_list = []
        mse_standard_list = []
        mse_supervised_list = []
        segrate_list = []

        for _ in tqdm(range(n_realizations), desc=f'Monte Carlo Simulation (T={T})'):
            r = generate_markov_process(T, P)
            Nsamples0, Nsamples1 = (r == 0).sum(), (r == 1).sum()
            s0 = generatesources(2, Nsamples0, 'rand')
            s1 = dependsourex1(Nsamples1)
            s = np.empty((2, T))
            s[:, r == 0] = s0
            s[:, r == 1] = s1

            # Mixing
            A = np.random.randn(2, 2)
            x = A @ s

            # Separation using ICA/ICE
            Bhat_ice, r_ice = icaice(Netats, x, 50, estim_eta=True, pihat_ini=np.array([0.5, 0.5]), nice=1, lambda_ICE=lambda_ICE)
            y_ice = Bhat_ice @ x
            mse_ice, _ = elimamb(y_ice, s)
            segrate_ice = (r_ice == r).sum() / T

            mse_ice_list.append(np.mean(mse_ice))
            segrate_list.append(segrate_ice)

            # Compare when ignoring latent model and dependence
            Bhat_standard = np.linalg.pinv(comica(x))
            y_standard = Bhat_standard @ x
            mse_standard, _ = elimamb(y_standard, s)
            mse_standard_list.append(np.mean(mse_standard))

            # Compare when supervised (known latent process r)
            Bhat_supervised = np.linalg.pinv(comica(x[:, r == 0]))
            y_supervised = Bhat_supervised @ x
            mse_supervised, _ = elimamb(y_supervised, s)
            mse_supervised_list.append(np.mean(mse_supervised))

        mse_ice_sorted_indices = np.argsort(mse_ice_list)
        mse_ice_sorted = np.array(mse_ice_list)[mse_ice_sorted_indices]
        segrate_sorted = np.array(segrate_list)[mse_ice_sorted_indices]

        mse_no_latent_sorted = np.sort(mse_standard_list)
        mse_supervised_sorted = np.sort(mse_supervised_list)

        # Plot MSE with logarithmic scale
        axes_mse[i].plot(range(n_realizations), mse_ice_sorted, 'b', label='MSE ICE')
        axes_mse[i].plot(range(n_realizations), mse_no_latent_sorted, 'r', label='MSE Standard')
        axes_mse[i].plot(range(n_realizations), mse_supervised_sorted, 'g', label='MSE Supervised')
        axes_mse[i].set_yscale('log')
        axes_mse[i].set_title(f'T={T} samples')
        axes_mse[i].set_xlabel('Realization Index')
        axes_mse[i].set_ylabel('MSE (log scale)')
        axes_mse[i].legend()

        # Plot Segmentation Rate
        axes_segrate[i].plot(range(n_realizations), segrate_sorted, 'g', label='Segmentation rate')
        axes_segrate[i].set_title(f'T={T} samples')
        axes_segrate[i].set_xlabel('Realization Index')
        axes_segrate[i].set_ylabel('Segmentation Rate')
        axes_segrate[i].legend()

    fig_mse.tight_layout()
    fig_mse.show()

    fig_segrate.tight_layout()
    fig_segrate.show()



def icaice(Netats,x, niter, estim_eta=True, pihat_ini=np.array([0.5, 0.5]), nice=1,
           lambda_ICE=5):
    """Combined ICA/ICE procedure as described in [1]

    Parameters
    ----------
    x : array_like, shape 2 x Nsamples
        mixed sources to be separated
    niter : int (default = 30)
        number of iterations for ICE
    estim_eta : bool, estimate probability for hidden state process.
        (default = True). If False, pihat_ini will be used.
    pihat_ini : 1D array
        initial probability values for hidden state process P(r) (default =
        uniform distribution)
    nice : int
        number of drawings for ICE stochastic approximation (default = 1)
    lambda_ICE : float (default = 5)
        parameter for dependent part of distribution

    Returns
    -------
    Bhat : array_like 2x2
        estimated separating matrix
    r_ice : array_like with bool 1xlen(x)
        last value of stochastic approximation of hidden process

    References
    ----------
    [1] Castella, M., Rafi, S., Comon, P., & Pieczynski, W., Separation of
        instantaneous mixtures of a particular set of dependent sources
        using classical ICA methods, EURASIP J. Adv. Signal Process., (62),
        (2013).
    """
    # initialization of the parameters
    _, T = x.shape
    pihat = pihat_ini
    Ahat = comica(x)
    Bhat = np.linalg.pinv(Ahat)  # np.linalg.pinv dans matlab -> inv?
    Phat = 0.5 * np.eye(Netats) + (np.ones((Netats, Netats)) - np.eye(Netats)) / (2 * (Netats - 1))
    Pxr = np.empty((2, T))
    for iter in range(niter):       # ICE iterations
        # --- calculate prob. conditionnally to r: Pxr(i,t) = Prob(x(t)/r(t)=i)
        y = Bhat@x

        #  conditionnally to r=0 (independent sources)
        # -- Prob(x(t)/r(t)=0) is assumed gaussian
        sigmatmp0 = 1
        Pxr[0, :] = 1/(2*np.pi*sigmatmp0**2)*np.exp(
            -(y**2).sum(axis=0)/(2*sigmatmp0**2))

        # conditionnally to r=1 (dependent sources)
        # -- along the 2 bisectors, Gaussian x Laplace
        rot = np.array([[1, 1], [1, -1]])/np.sqrt(2)
        z = rot@y
        lambdatmp1 = lambda_ICE
        sigmatmp1 = np.sqrt(2*(1-1/lambdatmp1**2))

        mulfac = 1/2*1/(sigmatmp1*np.sqrt(2*np.pi))*lambdatmp1/2
        Pxr[1, :] = mulfac*(np.exp(-z[0, :]**2/(2*sigmatmp1**2))
                            * np.exp(-lambdatmp1*abs(z[1, :]))
                            + np.exp(-z[1, :]**2/(2*sigmatmp1**2))
                            * np.exp(-lambdatmp1*abs(z[0, :])))

        alpha, beta, gamma, psi = forward_backward(T, 2, Pxr, pihat, Phat)

        pihat = np.sum(gamma, axis=1) / T

        denom = np.sum(gamma[:, :-1], axis=1, keepdims=True)
        denom[denom == 0] = 1  # Avoid division by zero
        Phat = np.sum(psi, axis=2) / denom

        r_ice = np.argmax(gamma, axis=0)
        # --- stochastic approximation of ICE ---
        Bhattmp = list()
        for iter in range(nice):
            Ahat = comica(x[:, r_ice==0])
            Bhattmp.append(np.linalg.inv(Ahat))
            Bhat = reduce(np.add, Bhattmp)/nice  #
    return Bhat, r_ice



if __name__ == '__main__':
    # generate data
    T = 5000                        # number of samples
    eta = 0.5        # eta = Prob(r=0) = Prob(s1 and s2 independent)
    Netats=2
    n_realizations = 100
    # generating the auxiliary process r (in this program, r is iid)
    P = np.array([[0.9, 0.1], [0.1, 0.9]])
    r = generate_markov_process(T, P)
    Nsamples0, Nsamples1 = (r==0).sum(), (r==1).sum() # keep parenthesis!

    s0 = generatesources(2, Nsamples0, 'rand')
    s1 = dependsourex1(Nsamples1)
    s = np.empty((2, T))
    s[:, r==0] = s0
    s[:, r==1] = s1
    # Mixing
    A = np.random.randn(2, 2)
    x = A@s

    # separation
    Bhat_ice, r_ice = icaice(Netats,x, 100, estim_eta=True,
                             pihat_ini=np.array([0.5, 0.5]), nice=1)
    # %-- final estimates --
    G_ice = Bhat_ice@A
    y = Bhat_ice@x
    mse_ice, _ = elimamb(y, s)
    segrate_ice = (r_ice == r).sum()/T
    # contestable, pourquoi pas pour N_ICE = 1

    #  -- display and print results --
    plt.ion()
    fig, ax = plt.subplots(num=1)
    ax.plot(s[0, r_ice==0], s[1, r_ice==0], 'b.')
    ax.plot(s[0, r_ice==1], s[1, r_ice==1], 'g.')
    # plt.show()
    print("--- Combined ICA/ICE results -----------")
    print(f"Segmentation rate: {segrate_ice}")
    print("Glogal mixing-separating matrix G "
          "(should be identity up to ambiguities)")
    print(G_ice)
    print("Separation criterion calculated on G:  "
          f"{1-(G_ice**2).max(axis=1)/(G_ice**2).sum(axis=1)}")
    print(f"MSE on each source {mse_ice}")

    print("--- Compare when ignoring latent model and dependence ---")
    Bhat = np.linalg.pinv(comica(x))
    Ghat = Bhat@A
    y = Bhat@x
    mse, _ = elimamb(y, s)
    print("Glogal mixing-separating matrix G "
          "(should be identity up to ambiguities)")
    print(Ghat)
    print("Separation criterion calculated on G:  "
          f"{1-(Ghat**2).max(axis=1)/(Ghat**2).sum(axis=1)}")
    print(f"MSE on each source {mse}")

    print("--- Compare when supervised (known latent process r) ---")
    Bhat_know = np.linalg.pinv(comica(x[:, r==0]))
    Ghat_know = Bhat_know@A
    y = Bhat_know@x
    mse_know, _ =elimamb(y, s)
    print("Glogal mixing-separating matrix G "
          "(should be identity up to ambiguities)")
    print(Ghat_know)
    print("Separation criterion calculated on G:  "
          f"{1-(Ghat_know**2).max(axis=1)/(Ghat_know**2).sum(axis=1)}")
    print(f"MSE on each source {mse_know}")


    #monte carlo similation
    #T_values=[1000,10000]
    #monte_carlo_simulation(Netats, n_realizations,T_values)