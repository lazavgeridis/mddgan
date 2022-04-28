import numpy as np
import torch
import scipy
from tensorly.tenalg import khatri_rao
from tensorly.decomposition import partial_tucker


def unfolding(n,A):
    shape = A.shape
    size = np.prod(shape)
    lsize = size // shape[n]
    sizelist = list(range(len(shape)))
    sizelist[n] = 0
    sizelist[0] = n
    return A.permute(sizelist).reshape(shape[n],lsize)


def modalsvd(n,A):
    nA = unfolding(n,A)
    return torch.svd(nA)


def hosvd(A):
    Ulist = []
    S = A
    for i,ni in enumerate(A.shape):
        u,_,_ = modalsvd(i,A)
        Ulist.append(u.numpy())
        assert not torch.any(torch.isnan(u)).item()
        assert not torch.any(torch.isnan(S)).item()
        S = torch.tensordot(S,u.t(),dims=([0],[0]))
        assert not torch.any(torch.isnan(S)).item()

    return S.numpy(),Ulist


def calq_cons(q, K):
    return q.reshape(*reversed(list(K)))


def svd(X):
    U, s, V = scipy.linalg.svd(X)
    return U, scipy.linalg.diagsvd(s, X.shape[0], X.shape[1]), V


def mdda(X, K, eps=1e-5):
    M = len(K)+1
    f = True
    d, N = X.shape
    assert np.prod(K) == d
    U, S, V = svd(X)
    B = U.dot(np.sqrt(S))
    Q = np.sqrt(S).dot(V)
    while f:
        As = []
        for i in range(N):
            Si, Ui = hosvd(torch.from_numpy(calq_cons(Q[:, i], K)))
            sigma = abs(Si[0, 0, 0])**(1/(M-1))
            if Si[0, 0, 0] < 0:
                sigma = -sigma
            for m in range(2, M+1):
                x = sigma*(Ui[M-m-1][0])
                if i == 0:
                    As.append(x[:, None])
                else:
                    As[m-2] = np.concatenate((As[m-2], x[:, None]), axis=1)
        katri = khatri_rao(As).T
        out = X.dot(katri)
        U, S, V = svd(out)
        B = U.dot(V.T)
        Q = B.T.dot(X)
        a = np.linalg.norm(X-B.dot(Q), ord='fro')**2
        b = np.linalg.norm(X, ord='fro')
        print(a/b, a, b, flush=True)
        f = a/b >= eps
    return B, As


def recomposition(B, As, coeffs):
    return B.dot(khatri_rao([c*a for a, c in zip(As, coeffs)]))


if __name__ == "__main__":
    m = 100
    d = 128
    B, As = mdda(np.random.randn(d, m), (4, 4, 8))
    data = recomposition(B, As, [0.1, 0.2, 0])

    import joblib
    W = joblib.load('matrix_w.pkl')
    B, As = mdda(W, (8, 8, 8))
