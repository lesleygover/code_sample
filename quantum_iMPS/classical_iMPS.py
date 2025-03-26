from scipy.stats import unitary_group
import numpy as np
from numpy.linalg import eig
from ncon import ncon
from math import floor
import cirq

from quten.circuits import StateAnsatzReducedQscout
"""
Contains code to work with classical iMPS, including generating them, calculating environments and transfer matrices and embedding them in a unitary
"""
def generate_random_state_left(d, D):
    """
    Generate a random state tensor A (σ, Dl, Dr) where σ = d and Dl = Dr = D with left canoncalisation.
    """
    U = unitary_group.rvs(d*D)
    U = U.reshape(d, D, d, D)
    U = U.transpose(2, 3, 0, 1)
    zero_state = np.eye(d)[0, :]

    A = ncon([U, zero_state], ((-2, -1, 1, -3), (1,)))
    return A

def generate_random_operator_left(d, D):
    """
    Generate a random operator tensor W (σ, Dl, l, Dr) = (d, D, d, D) with the
    left canonicalisation as outlined by:

       |
    -- O  --
    |  |         ==  I
    -- O* --
       |
    """
    W = unitary_group.rvs(d*D)
    W = W.reshape(d, D, d, D)
    #W = W.transpose(0, 2, 1, 3)
    return W

def merge_AB(A, B):
    '''
    Merge two state tensors together as shown:

    -- A -- B --
       |    |

    Args
    ----
    A, B : Arrays of shape (σ, Dl, Dr)
    '''
    la, σa,  _ = A.shape
    _, σb, rb = B.shape
    AB = ncon([A, B], ((-1, -2, 1), (1, -3, -4)))
    AB = AB.reshape(la, σa*σb, rb)
    return AB

def map_AB(A, B):
    '''
    Combine A, B as follows
    i -- A -- j    ,   k -- B -- l
         |                  |
    =
    i -- A -- j
         |
    k -- B -- l
    where the shape of the output is (i*k, j*l)

    Args
    ----
    A, B: State tensors with shape (Dli, σi, Dri) where i is A and B respectively.
    '''

    la, _, ra = A.shape
    lb, _, rb = B.shape

    M = ncon([A, B.conj()], ((-1, 1, -3), (-2, 1, -4)))
    M = M.reshape(la*lb, ra*rb)
    return M

def right_fixed_point(E, all_evals=False):
    '''
    Calculate the right fixed point of a transfer matrix E

    E.shape = (N, N)
    '''
    evals, evecs = eig(E)
    sort = sorted(zip(evals, evecs), key=lambda x: np.linalg.norm(x[0]),
                   reverse=True)
    # Look into `scipy.sparse.linalg.eigs, may be faster`
    if all_evals:
        mu, r = list(zip(*sort))
        return np.array(mu), np.array(r)
    mu, r = sort[0]
    return mu, r


def fidelityDensity(A, B):
    TM = map_AB(A, B)

    mu, _ = right_fixed_point(TM)
    return np.abs(mu)


def map_AWB(A, W, B):
    '''
    Combine A, B states with an operator W in the middle as follows
    i -- A -- j
         |
    k -- W -- l
         |
    m -- B -- n
    where the shape of the output is (i*k*m, j*l*n)

    Args
    ----
    A, B: State tensors with shape (σi, Dli, Dri) where i is A and B respectively.
    W   : Operator tensor with shape (σ, Dl, l, Dr)  (σ index connects to B and l index connects to A).
    '''
    la, _, ra = A.shape
    lb, _, rb = B.shape
    _, lw, _, rw = W.shape

    M = ncon([A, W, B.conj()], ((-1, 1, -4), (1, -2, 2, -5), (-3, 2, -6)))
    M = M.reshape(la*lb*lw, ra*rb*rw)
    return M

def generate_right_environment(transferMatrix):
    """
    For a given transfer matrix produce the vector that gives the right
    environment.

    Args
    ----
    transferMatrix : (N, N) array representing transfer matrix.
    """
    evals, evecs = eig(transferMatrix)
    magEvals = np.abs(evals)
    maxArg = np.argmax(magEvals)

    R = evecs[:, maxArg]
    return R

def embed_state_in_unitary(ψ):
    """
    Embed a state ψ into a unitary V in the |0> state such that V|0> = ψ.
    Note this requires the <0|ψ> term in the vector to be real.
    """
    dim = ψ.shape[0]
    zero = np.eye(dim)[0]
    assert np.isclose(np.imag(np.dot(zero, ψ)), 0.0), "First element of ψ needs to be real"
    v = (zero + ψ) / np.sqrt(2*(1 + np.dot(zero, ψ)))
    return 2*np.outer(v, v.conj()) - np.eye(dim)


def unitary_to_tensor_left(U):
    '''
    Take a unitary U and make it a left canonical tensor A such that
   |0>     k
    |     |
    |     |     |
    ---U---     | direction of unitary
    |     |     |
    |     |     v
    i     j

    Note `U.shape = (i j, |0> k)` and which generates state tensor `A.shape = (i, j, k)`

    i == A == k
         |
         j
    '''
    n = int(np.log2(U.shape[0]))
    zero = np.array([1., 0.])

    Ucontr = list(range(-1, -n-1, -1)) + [1] + list(range(-n-1, -2*n, -1))
    A = ncon([U.reshape(*2 * n *[2]), zero], [Ucontr, [1,]])
    A = A.reshape(2**(n-1), 2, 2**(n-1))
    return A


def expectation(params1,params2, Op):
    A = unitary_to_tensor_left(cirq.unitary(StateAnsatzReducedQscout(params1)))
    B = unitary_to_tensor_left(cirq.unitary(StateAnsatzReducedQscout(params2)))
    TM = map_AWB(A, Op, B)
    mu, _ = right_fixed_point(TM)
    return np.abs(mu)**2


def losch_overlap(params1,params2):
    A = unitary_to_tensor_left(cirq.unitary(StateAnsatzReducedQscout(params1)))
    B = unitary_to_tensor_left(cirq.unitary(StateAnsatzReducedQscout(params2)))
    overlap = fidelityDensity(A,B)
    return -np.log(overlap**2)