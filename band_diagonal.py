import numpy as np

def metropolis(e1, e2, t):
    """
    Evaluate metropolis condition with:
    
      e1 :  Energy function for current state
      e2 :  Energy function for proposed state
      t  :  temperature
      
    Always swap if new state is better.  Take disadvantageous swap with probability
    p = exp(-t*(e2-e1)/tau) (probability decreases exponentially with increasing 
    iterations - "cooling" in simulated annealing)
    """
    if e2 < e1:
        return True
    
    x = (e2-e1)/t
    
    if np.random.rand() < np.exp(-x):
        return True
    return False


def simulated_annealing(x0, energy_fun, perm_fun, 
                        steps=10000, t0=1.0, tau=1,
                        report_steps=100,
                        report=False):
    x = x0.copy()
    e = np.zeros(shape=(steps,))
    swaps = list()
    for t in np.arange(0, steps):
        x_cand = x.copy()
        perm_fun(x_cand)
        e1 = energy_fun(x)
        e2 = energy_fun(x_cand)
        temp = tau/t
        swap = metropolis(e1, e2, temp)
        if swap:
            x = x_cand
            e[t] = e2
        else:
            e[t] = e1
        swaps.append(swap)
            
        if report and t % report_steps == 0:
            print('[{0}] Energy = {1}, T={2}'.format(t, e[t], temp))
        
    return x, e, swaps


def permute_two(mat, i1, i2):
    """
    Re-order matrix by swapping two rows / columns
    """
    if i1 == i2:
        return
    
    # Row exchange
    row1 = np.array(mat[i1, :])
    row2 = np.array(mat[i2, :])
    mat[i1, :] = row2
    mat[i2, :] = row1
    
    # Column exchange
    col1 = np.array(mat[:, i1])
    col2 = np.array(mat[:, i2])
    mat[:, i1] = col2
    mat[:, i2] = col1
    

def random_permute_two(mat):
    i1 = np.random.randint(0, mat.shape[0])
    i2 = np.random.randint(0, mat.shape[0])
    permute_two(mat, i1, i2)
    return i1, i2
    
def random_permutation_matrix(n):
    indices = np.arange(0,n)
    shuffled = np.random.permutation(indices)
    
    mat = np.zeros(shape=(n,n))
    
    for i in indices:
        j = shuffled[i]
        mat[i,j] = 1
    
    return shuffled, mat


class ReorderPermuter:
    """
    This is just a wrapper class for the random row/column swap function that does additional 
    book-keeping on the index array.  After the algorithm is complete, the index array will have been
    re-ordered to provide lookup of labels associated with the matrix or to generate the permutation
    matrix solved for by simulated annealing.
    """
    
    def __init__(self, original_mat, indices=None, n=None):
        if indices is None:
            if n is None:
                raise Exception('Must supply either array of indices or value of n (matrix dimension)')
            indices = np.arange(0, n)
            n = indices.shape[0]
        self.n = n
        self.indices = indices
        self.original_mat = original_mat.copy()
        
    def random_row_column_reorder(self, mat):
        i1, i2 = random_permute_two(mat)
        # swap indices
        tmp = self.indices[i2]
        self.indices[i2] = self.indices[i1]
        self.indices[i1] = tmp
        
    def get_permutation_matrix(self):
        
        permutation_mat = np.zeros(shape=(self.n, self.n))
        for i in np.arange(0, self.n):
            permutation_mat[i, self.indices[i]] = 1.0
            
    def reorder_with_permutation(self, mat):
        
        pmat = get_permutation_matrix()
        # Row / column reorder with permutation matrix
        return np.matmul(pmat.transpose(), np.matmul(mat, pmat))


def rbf_cov(n, length_scale=None):
    """
    Simple rbf kernel covariance with "distance" uniformly tied to
    index in the matrix (creates band-diagonal matrix for testing)
    """
    if length_scale is None:
        length_scale = n/4
    scale = np.power(length_scale, 2)
    cov = np.zeros(shape=(n, n))
    for i in np.arange(0, n):
        for j in np.arange(0, n):
            d = np.power(i-j, 2) / scale
            cov[i,j] = np.exp(-d)
    return cov


def matrix_score(m):
    """
    Objective function to penalize off-diagonal matrix mass.  Yields 
    zero for a diagonal matrix.  Minimization of this function over 
    row/column ordering minimizes matrix bandwidth (akin to banded 
    """
    score = 0
    m_abs = np.abs(m)
    norm = np.power(m.shape[0], 4) / 2
    for i in np.arange(0, m.shape[0]):
        for j in np.arange(0, m.shape[1]):
            score += m_abs[i,j]*(i-j)*(i-j)
            
    return score / norm
