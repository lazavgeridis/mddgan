import numpy as np
import scipy
import joblib
from tensorly.tenalg import khatri_rao
from tensorly.base import unfold


def hosvd(A):
    """Multilinear SVD
        
       Parameters
       ----------
       A : numpy.ndarray
            tensor Q_i of shape (K_M, K_M-1, ... , K2)
       """
    #print(A.ndim)
    U = []
    S = A
    for dim in range(A.ndim):
        flat = unfold(A, dim)
        U_m, s, v = svd(flat)
        U_m = U_m[:, 0]
        U.append(U_m)
        s = s[0, 0]
        v = v[:, 0]
        S = np.tensordot(S, U_m.conj().T, axes=([0], [0]))

    return S, U


def calq_cons(q_i, sizes):
    return q_i.reshape(*reversed(list(sizes)))


def svd(X):
    #print(X.shape)
    U, s, V = scipy.linalg.svd(X)
    return U, scipy.linalg.diagsvd(s, X.shape[0], X.shape[1]), V


def mdda(X, *sizes, eps=1e-6):
    print('Modes of variation: ', sizes)
    M = len(sizes) + 1 # modes of variation
    #M = len(sizes)   # modes of variation
    cond = True
    d, N = X.shape
    assert np.prod(sizes) == d
    U, S, V = svd(X)
    B = U.dot(np.sqrt(S))
    Q = np.sqrt(S).dot(V.T)
    variation_coefs = [[] for _ in range(M - 1)]
    while cond:
        for i in range(N):
        #for i in range(1):
            Si, Ui = hosvd(calq_cons(Q[:, i], sizes))   # s_i is a scalar
            sigma = abs(Si)**(1 / (M - 1))
            #sigma = abs(Si[0, 0, 0])**(1 / (M - 1)) # is abs() really needed???
            #if Si[0, 0, 0] < 0:
            #    sigma = -sigma
            for m in range(2, M + 1):
                x = sigma * (Ui[M - m - 1]).reshape(-1, 1)
                #x = sigma * (Ui[M - m - 1])
                #if i == 0:
                #    As.append(x[:, None])
                #else:
                #    As[m-2] = np.concatenate((As[m-2], x[:, None]), axis=1)
                variation_coefs[m - 2].append(x)

        variation_modes = [np.concatenate(coefs, axis=1) for coefs in variation_coefs]
        out = X.dot(khatri_rao(variation_modes).T)
        print(out.shape)
        U, S, V = svd(out)
        B = U.dot(V.T)
        Q = B.T.dot(X)
        a = np.linalg.norm(X-B.dot(Q), ord='fro')**2
        b = np.linalg.norm(X, ord='fro')**2
        print(a/b, a, b, flush=True)
        cond = (a/b >= eps)
        for coefs in variation_coefs:
            coefs.clear()
    return B, variation_modes


if __name__ == "__main__":
    #m = 100
    #d = 128
    #B, As = mdda(np.random.randn(d, m), (4, 4, 8))
    #data = recomposition(B, As, [0.1, 0.2, 0])

    # CELEBA-HQ has 40 binary attributes
    # e.g male or female face
    W = joblib.load('./celebahq1024_stylemodW.pkl')
    print('Weights with shape: ', W.shape)
    # choose one of the above
    #dimensions = [8, 16, 4]
    #dimensions = [8, 8, 8]
    dimensions = [8, 32, 2]
    B, As = mdda(W, dimensions[0], dimensions[1], dimensions[2])
    print(B.shape)
    for A in As:
        print(A.shape)
    joblib.dump(B, 'B_1.pkl')
