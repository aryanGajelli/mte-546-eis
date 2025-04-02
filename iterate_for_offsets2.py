import numpy as np
from multiprocessing import Pool, cpu_count

# Define constants
f = np.geomspace(0.1, 1, 10)
# Cut off the second half frequencies
f = f[:5]
t = np.arange(0, 40, 0.01, dtype=np.float64)
omega = 2 * np.pi * f[:, None]

def evaluate_phi(phi):
    s = np.sum(np.sin(omega * t + phi[:, None]), axis=0)
    return np.max(np.abs(s))

def random_search_chunk(chunk_size):
    np.random.seed()
    phis = np.random.uniform(0, 2 * np.pi, (chunk_size, 5)).astype(np.float64)
    best_M_local = float('inf')
    best_phi_local = None
    for phi in phis:
        M = evaluate_phi(phi)
        if M < best_M_local:
            best_M_local = M
            best_phi_local = phi.copy()
    return best_M_local, best_phi_local

if __name__ == '__main__':
    n_iter = 10_000_000
    n_cores = cpu_count()
    chunk_size = n_iter // n_cores

    with Pool(n_cores) as pool:
        results = pool.map(random_search_chunk, [chunk_size] * n_cores)

    M_values, phi_values = zip(*results)
    best_idx = np.argmin(M_values)
    best_M = M_values[best_idx]
    best_phi = phi_values[best_idx]

    print(f"Best peak magnitude: {best_M:.6f}")
    print(f"Optimal phases: {best_phi}")