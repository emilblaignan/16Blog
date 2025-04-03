import numpy as np
import jax.numpy as jnp
import jax
from scipy.sparse import diags
from jax.experimental.sparse import BCOO

def advance_time_matvecmul(A, u, epsilon):
    """Advances the simulation by one timestep, via matrix-vector multiplication
    Args:
        A: The 2d finite difference matrix, N^2 x N^2. 
        u: N x N grid state at timestep k.
        epsilon: stability constant.

    Returns:
        N x N Grid state at timestep k+1.
    """
    N = u.shape[0]
    u = u + epsilon * (A @ u.flatten()).reshape((N, N))
    return u

def get_A(N):
    """Constructs the finite difference matrix A for the heat equation.
    Args:
        N: Grid size

    Returns:
        A: Finite difference matrix (N^2 x N^2).
    """
    n = N * N
    diagonals = [
        -4 * np.ones(n),  # Main diagonal
        np.ones(n - 1),    # Right neighbor
        np.ones(n - 1),    # Left neighbor
        np.ones(n - N),    # Upper neighbor
        np.ones(n - N)     # Lower neighbor
    ]
    # Apply boundary conditions (preventing wrap-around)
    diagonals[1][(N-1)::N] = 0
    diagonals[2][(N-1)::N] = 0  

    # Construct the finite difference matrix
    A = (
        np.diag(diagonals[0]) +
        np.diag(diagonals[1], 1) +
        np.diag(diagonals[2], -1) +
        np.diag(diagonals[3], N) +
        np.diag(diagonals[4], -N)
    )
    return A

def get_sparse_A(N):
    """Constructs the sparse matrix A using JAX sparse format."""
    A = get_A(N) # Get the dense matrix A
    A_sp_matrix = BCOO.fromdense(jnp.array(A)) # Convert dense matrix to JAX sparse format (BCOO)
    return A_sp_matrix

def advance_time_numpy(u, epsilon):
    """Advances the simulation using NumPy's np.roll for boundary handling."""
    u_padded = np.pad(u, pad_width=1, mode='constant', constant_values=0)  # Pad u with zeros
    u_next = u + epsilon * (  # Update u using finite differences
        np.roll(u_padded, 1, axis=0)[1:-1, 1:-1] +  # Shift up
        np.roll(u_padded, -1, axis=0)[1:-1, 1:-1] +  # Shift down
        np.roll(u_padded, 1, axis=1)[1:-1, 1:-1] +  # Shift left
        np.roll(u_padded, -1, axis=1)[1:-1, 1:-1] -  # Shift right
        4 * u  # Subtract central value
    )
    return u_next  # Return updated interior values

@jax.jit
def advance_time_jax(u, epsilon):
    """Advances the simulation using JAX and just-in-time compilation."""
    u_padded = np.pad(u, pad_width=1, mode='constant', constant_values=0)  # Pad u with zeros
    u_next = u + epsilon * (  # Update u using finite differences
        jnp.roll(u_padded, 1, axis=0)[1:-1, 1:-1] +  # Shift up
        jnp.roll(u_padded, -1, axis=0)[1:-1, 1:-1] +  # Shift down
        jnp.roll(u_padded, 1, axis=1)[1:-1, 1:-1] +  # Shift left
        jnp.roll(u_padded, -1, axis=1)[1:-1, 1:-1] -  # Shift right
        4 * u  # Subtract central value
    )
    return u_next  # Return updated interior values
