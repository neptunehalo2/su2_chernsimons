import numpy as np

# Chern-Simons constant
k = 1.0

# Grid dimensions (lattice units)
Lx, Ly, Lz = 50, 50, 50  # 3D grid size

# Lattice spacing
dx = dy = dz = 1.0

# Pauli matrices (generators of SU(2))
Id2=np.array([[1, 0], [0, 1]])  # Identity 2x2
sigma_1 = np.array([[0, 1], [1, 0]])  # Pauli X
sigma_2 = np.array([[0, -1j], [1j, 0]])  # Pauli Y
sigma_3 = np.array([[1, 0], [0, -1]])  # Pauli Z

# List of generators
sigma = [sigma_1, sigma_2, sigma_3]

import numpy as np

# Function to create a random matrix in the SU(2) algebra using the quaternion statement
def create_random_SU2_from_quaternion():
    # Generate random numbers for the quaterion formula
    a, b, c, d = np.random.randn(4)
    
    # Nomralize the quaternion such a^2 + b^2 + c^2 + d^2 = 1
    norm = np.sqrt(a**2 + b**2 + c**2 + d**2)
    a, b, c, d = a / norm, b / norm, c / norm, d / norm
    
    # Build the SU(2) matrix
    U = np.array([[a + 1j*b, c + 1j*d],
                  [-c + 1j*d, a - 1j*b]])
    
    return U

# Function to create a random SU(2) gauge field
def create_gauge_field(Lx, Ly, Lz):
    # Initialize a 2x2 matrix for each grid point
    A = np.zeros((Lx, Ly, Lz, 2, 2), dtype=complex)
    
    for i in range(Lx):
        for j in range(Ly):
            for k in range(Lz):

                # Store the gauge field at point (i, j, k)
                A[i, j, k] = create_random_SU2_from_quaternion()

    return A

# Function to calculate the field strength F_{\mu\nu}
def calculate_field_strength(A, dx, dy, dz):
    Lx, Ly, Lz, _, _ = A.shape
    F = np.zeros((Lx, Ly, Lz, 3, 3, 2, 2), dtype=complex)  # Store F_{\mu\nu}
    
    for i in range(Lx):
        for j in range(Ly):
            for k in range(Lz):
                # Derivatives in x, y, z directions
                for mu in range(3):
                    for nu in range(mu, 3):  # Only calculate for upper triangular part (due to symmetry)
                        if mu == 0:  # Derivative in x-direction
                            dA_mu = (A[i+1, j, k] - A[i, j, k]) / dx if i < Lx-1 else (A[i, j, k] - A[i-1, j, k]) / dx
                        elif mu == 1:  # Derivative in y-direction
                            dA_mu = (A[i, j+1, k] - A[i, j, k]) / dy if j < Ly-1 else (A[i, j, k] - A[i, j-1, k]) / dy
                        else:  # Derivative in z-direction
                            dA_mu = (A[i, j, k+1] - A[i, j, k]) / dz if k < Lz-1 else (A[i, j, k] - A[i, j, k-1]) / dz
                            
                        if nu == 0:  # Derivative in x-direction
                            dA_nu = (A[i+1, j, k] - A[i, j, k]) / dx if i < Lx-1 else (A[i, j, k] - A[i-1, j, k]) / dx
                        elif nu == 1:  # Derivative in y-direction
                            dA_nu = (A[i, j+1, k] - A[i, j, k]) / dy if j < Ly-1 else (A[i, j, k] - A[i, j-1, k]) / dy
                        else:  # Derivative in z-direction
                            dA_nu = (A[i, j, k+1] - A[i, j, k]) / dz if k < Lz-1 else (A[i, j, k] - A[i, j, k-1]) / dz
                        
                        # Commutator of the gauge field components
                        commutator = np.dot(A[i, j, k], A[i, j, k]) - np.dot(A[i, j, k], A[i, j, k].T)
                        
                        # Field strength F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu + [A_\mu, A_\nu]
                        F[i, j, k, mu, nu] = dA_mu - dA_nu + commutator
    
    return F

# Function to calculate the Chern-Simons action
def chern_simons_action(A, dx, dy, dz, k):
    Lx, Ly, Lz, _, _ = A.shape
    action = 0.0  # Initialize the action
    
    F = calculate_field_strength(A, dx, dy, dz)
    
    for i in range(Lx):
        for j in range(Ly):
            for k in range(Lz):
                # Compute the action S_CS = k / (4*pi) * epsilon^{mu nu lambda} Tr(A_mu F_{nu lambda})
                epsilon = np.zeros((3, 3, 3), dtype=int)
                epsilon[0, 1, 2] = 1
                epsilon[0, 2, 1] = -1
                epsilon[1, 0, 2] = -1
                epsilon[1, 2, 0] = 1
                epsilon[2, 0, 1] = 1
                epsilon[2, 1, 0] = -1
                
                # Calculate the contribution from epsilon^{mu nu lambda} Tr(A_mu F_{nu lambda})
                for mu in range(3):
                    for nu in range(3):
                        for lambda_ in range(3):
                            contribution = epsilon[mu, nu, lambda_] * np.trace(np.dot(A[i, j, k], F[i, j, k, mu, nu]))
                            action += contribution
    
    # Normalize by the factor k/(4*pi)
    action *= k / (4 * np.pi)
    
    return action

# Main function
def main():
    avg_action=0
    for i in range(50):
        # Create a random SU(2) gauge field
        A = create_gauge_field(Lx, Ly, Lz)
        
        # Calculate the Chern-Simons action
        action = chern_simons_action(A, dx, dy, dz, k)
        avg_action+=action
        # Print the action
        print(f"Chern-Simons action: {avg_action}")
    print(f"AverageChern-Simons action: {avg_action/50}")

# Run the script
if __name__ == "__main__":
    main()
