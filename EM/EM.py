import numpy as np
from scipy import signal

def convolve(X, kernel, mode='valid'):
    return signal.fftconvolve(X, kernel, mode=mode, axes=(0, 1))

def conv_norm(X, F, B):
    F_norm = np.linalg.norm(F) ** 2
    
    kernel = np.ones_like(F)[..., None]
    B_ = B[..., None]
    X_B_norm = np.linalg.norm(X - B_, axis=(0, 1)) ** 2
    X_B_conv = convolve((2 * X - B_) * B_, kernel)
    X_F_conv = convolve(X, F[::-1, ::-1, None])
    
    norm = X_B_conv - 2 * X_F_conv + F_norm + X_B_norm
    return np.where(norm < 0, 0, norm)

def calculate_log_probability(X, F, B, s):
    H, W, K = X.shape
    h, w = F.shape
    norm_const = H * W * (np.log(2 * np.pi) / 2 + np.log(s))
    norm = conv_norm(X, F, B)
    ll = - norm_const - norm / (2 * s ** 2)
    return ll

def calculate_lower_bound(X, F, B, s, A, q, use_MAP=False):
    ll = calculate_log_probability(X, F, B, s)

    if use_MAP:
        K = X.shape[-1]
        Q = ll[q[0], q[1], np.arange(K)] + np.log(A + 1e-15)[q[0], q[1]]
        L = np.sum(Q)
    else:
        Q = ll + np.log(A + 1e-15)[..., None] - np.log(q + 1e-15)
        L = np.sum(q * Q)

    return L

def run_e_step(X, F, B, s, A, use_MAP=False):
    ll = calculate_log_probability(X, F, B, s)
    ll_max = ll - ll.max((0, 1))
    q = np.exp(ll_max) * A[..., None]
    
    if use_MAP:
        q_h, q_w, K = q.shape
        q_map = q.swapaxes(0, 2).swapaxes(1, 2).reshape((K, -1))
        q_map = np.argmax(q_map, axis=-1)
        
        q = np.vstack(np.unravel_index(q_map, (q_h, q_w)))
    else:
        q /= q.sum((0, 1))
        
    return q

def run_m_step(X, q, h, w, use_MAP=False):
    H, W, K = np.shape(X)
    
    if use_MAP:
        F = np.zeros((h, w))
        B = np.zeros((H, W))
        B_count = np.ones((H, W)) * K
        A = np.zeros((H-h+1, W-w+1))
        
        for k in range(K):
            i, j = q[0, k], q[1, k]
            F += X[i:i+h, j:j+w, k] 
            B_k = X[..., k].copy()
            B_k[i:i+h, j:j+w] = 0
            B += B_k
            B_count[i:i+h, j:j+w] -= 1
            A[i, j] += 1
                                
        F /= K
        B_count[B_count == 0] = 1
        B /= B_count
        A /= K
        
        Y = np.ones_like(X) * B[..., None]
        for k in range(K):
            i, j = q[0, k], q[1, k]
            Y[i:i+h, j:j+w, k] = F.copy()
        
        s = np.linalg.norm(X - Y) ** 2
        s /= (K * H * W)
        
    else:
        kernel = np.ones((h, w, K))    
        Q = 1 - convolve(q, kernel, mode='full')
        B = np.average(X, weights=Q, axis=-1)
        F = np.mean(convolve(X, q[::-1, ::-1]), axis=-1)

        A = np.mean(q, axis=-1)
        norm = conv_norm(X, F, B)
        s = np.sum(q * norm) / (K * H * W)
    
    s = np.sqrt(s)
    return F, B, s, A

def run_EM(X, h, w, F=None, B=None, s=None, A=None, tolerance=0.001,
           max_iter=50, use_MAP=False):
    
    H, W, K = X.shape
    x_max = X.max()
    F = x_max * np.random.rand(h, w) if F is None else F
    B = x_max * np.random.rand(H, W) if B is None else B
    s = 1 - np.random.rand() if s is None else s
    A = np.random.rand(H-h+1, W-w+1) if A is None else A
    A /= A.sum()
    
    LL = []
    
    for i in range(max_iter):
        q = run_e_step(X, F, B, s, A, use_MAP=use_MAP)
        F, B, s, A = run_m_step(X, q, h, w, use_MAP=use_MAP)
        L = calculate_lower_bound(X, F, B, s, A, q, use_MAP=use_MAP)
        LL.append(L)
        if i and LL[-1] - LL[-2] < tolerance:
            break
            
    return F, B, s, A, LL

def run_EM_with_restarts(X, h, w, tolerance=0.001, max_iter=50, use_MAP=False,
                         n_restarts=10):
    for restart in range(n_restarts):
        F, B, s, A, LL = run_EM(X, h, w, tolerance=tolerance, max_iter=max_iter, use_MAP=use_MAP)

    return F, B, s, A, LL 

