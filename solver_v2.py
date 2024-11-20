import sys
import typing
import scipy as sp # This shouldn't have to be in here, it breaks minimal imports

import autograd.numpy as npa
import numpy as np
import numpy.typing as npt


# Everything in this file is set up for a single dimension of movement. It must be altered to allow for a second dimension.

def quar(mat: npt.ArrayLike) -> np.ndarray:
    assert mat.shape[0] == mat.shape[1]
    assert mat.shape[0] % 2 == 0
    mid = int(mat.shape[0] / 2)
    return np.array([mat[:mid, :mid], mat[:mid, mid:], mat[mid:, :mid], mat[mid:, mid:]])


def manual_matmul(A, B, threshold = None, dtype = None):  # TODO vectorize for GPU use and give errors
    if dtype is None:
        dtype = np.cdouble
    else:
        dtype = np.dtype(dtype)
    """
    Manually performs matrix multiplication between two NumPy arrays A and B.

    Parameters:
    - A: NumPy array of shape (m, n)
    - B: NumPy array of shape (n, p)

    Returns:
    - result: NumPy array of shape (m, p) resulting from A x B
    """

    # Get the dimensions of the input matrices
    a_rows, a_cols = A.shape
    b_rows, b_cols = B.shape

    # Check if the matrices can be multiplied
    if a_cols != b_rows:
        raise ValueError("Incompatible dimensions for matrix multiplication.")

    # Initialize the result matrix with zeros
    result = np.zeros((a_rows, b_cols), dtype=dtype) # This class has worked so far to elim half-zero errors

    # Perform the matrix multiplication manually
    for i in range(a_rows):
        for j in range(b_cols):
            for k in range(a_cols):  # or range(b_rows)
                result[i, j] += A[i, k] * B[k, j]
    if threshold is not None:
        result[np.abs(result) < threshold] *= 0
    return result


def redheffer_star_product(
    a: npt.ArrayLike, b: npt.ArrayLike
) -> np.ndarray:
    """
    This is a function that takes two matrices and returns their Redheffer star product.
    """
    assert (
        a.shape[0] % 2 == 0
        and a.shape[1] % 2 == 0
        and b.shape[0] % 2 == 0
        and b.shape[1] % 2 == 0
    ), "Redheffer star product only works for even matrices"
    a00, a01, a10, a11 = (
        a[: a.shape[0] // 2, : a.shape[1] // 2],
        a[: a.shape[0] // 2, a.shape[1] // 2:],
        a[a.shape[0] // 2:, : a.shape[1] // 2],
        a[a.shape[0] // 2:, a.shape[1] // 2:],
    )
    b00, b01, b10, b11 = (
        b[: b.shape[0] // 2, : b.shape[1] // 2],
        b[: b.shape[0] // 2, b.shape[1] // 2:],
        b[b.shape[0] // 2:, : b.shape[1] // 2],
        b[b.shape[0] // 2:, b.shape[1] // 2:],
    )
    q = a00.shape[0]
    assert (
        np.linalg.det(np.eye(q) - a01 @ b10) != 0
    ), "Redheffer star product only works for matrices with invertible a01@b10"
    s00 = b00 @ np.linalg.inv(np.eye(q) - a01 @ b10) @ a00
    s01 = b01 + b00 @ np.linalg.inv(np.eye(q) - a01 @ b10) @ a01 @ b11
    s10 = a10 + a11 @ np.linalg.inv(np.eye(q) - b10 @ a01) @ b10 @ a00
    s11 = a11 @ np.linalg.inv(np.eye(q) - b10 @ a01) @ b11
    return np.block([[s00, s01], [s10, s11]])


def extended_redheffer_star_product(mat1: npt.ArrayLike, mat2: npt.ArrayLike) -> np.ndarray:
    '''
    Assumes buffer between two systems.
    Compute the system scattering matrix and take the input from the bottom to be 0. Then use the reflected and the inputs to calculate backwards (while moving around the matrices) what the next coupling coefficients should be.
    '''
    sa00, sa01, sa10, sa11 = quar(mat1)
    sb00, sb01, sb10, sb11 = quar(mat2)
    identity = np.eye(s00.shape[0])
    # This matrix accounts for the infinite reflections (geometric) that occur at each interface
    inf1 = np.linalg.inv(identity - sb00 @ sa11)
    inf2 = np.linalg.inv(identity - sa11 @ sb00)
    '''
    Assuming that each scattering matrix has been computed (shown here), then the updates to the mode coefficients are simply those modes * the scattering matrices themselves.
    '''
    sc00 = sa00 + sa01@inf1@sb00@sa10
    sc01 = sa01@inf1@sb01
    sc10 = sb10@inf2@sa10
    sc11 = sb11 + sb10@inf2@sa11@sb01
    sc = np.block([[sc00, sc01], [sc10, sc11]])
    return sc


class Layer_:
    """
    Class for defining a single layer of a layer stack used in a simulation
    """

    def __init__(
        self,
        permeability: npt.ArrayLike,
        permittivity: npt.ArrayLike,
        thickness: float,
        is_incident_layer: bool = False,
        n_harmonics: int = 1,
    ):
        """
        No crystal should be created beforehand. We interpret everything the same way so debugging is easier.
        """
        self.permittivity = permittivity
        self.permeability = permeability
        self.thickness = thickness
        # This may become useful depending on how the calculations are implemented
        self.is_incident_layer = is_incident_layer
        self.n_harmonics = n_harmonics

    def layer_distribution_convolution_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        permittivity_convolution_matrix = np.zeros(
            (self.n_harmonics * 2 + 1, self.n_harmonics * 2 + 1), dtype=np.cdouble
        )
        permeability_convolution_matrix = np.zeros(
            (self.n_harmonics * 2 + 1, self.n_harmonics * 2 + 1), dtype=np.cdouble
        )  # This expands to n_harmonics ^ 2 X n_harmonics ^ 2 in the 2d case

        permittivity_fourier_representation = np.fft.fftshift(
            np.fft.fft(self.permittivity, axis=0)
        )
        permeability_fourier_representation = np.fft.fftshift(
            np.fft.fft(self.permeability, axis=0)
        )

        zero_harmonic = len(self.permittivity) // 2
        (
            permittivity_fourier_representation,
            permeability_fourier_representation,
        ) = permittivity_fourier_representation[
            zero_harmonic
            - 2 * self.n_harmonics: zero_harmonic
            + 2 * self.n_harmonics
            + 1
        ] / len(
            permittivity_fourier_representation
        ), permeability_fourier_representation[
            zero_harmonic
            - 2 * self.n_harmonics: zero_harmonic
            + 2 * self.n_harmonics
            + 1
        ] / len(
            permeability_fourier_representation
        )
        # For a single harmonic, we can use Toeplitz symmetry (as we derived) to obtain the resulting matrix
        # +1 to account for the zero harmonic
        for x in range(2 * self.n_harmonics + 1):
            for y in range(2 * self.n_harmonics + 1):
                permittivity_convolution_matrix[y, x] = (
                    permittivity_fourier_representation[y -
                                                        x + 2 * self.n_harmonics]
                )
                permeability_convolution_matrix[y, x] = (
                    permeability_fourier_representation[y -
                                                        x + 2 * self.n_harmonics]
                )
        return permittivity_convolution_matrix, permeability_convolution_matrix


class Solver_:
    """
    Solver class for solving the RCWA equations.
    Each one of these things is stored in a dictionary.
    """

    def __init__(
        self,
        layer_stack: typing.List[Layer_],
        grating_period: float,  # Period of the grating in the x-direction.
        wavelength: float,
        n_harmonics: int = 0,
        theta: float = 0,
    ):
        '''
        assert layer_stack.len > 2
        '''
        self.internal_layer_stack = layer_stack[1:-1]
        self.top_layer = layer_stack[0]  # Semi-infinite
        self.bottom_layer = layer_stack[-1]  # Semi-infinite
        self.n_harmonics = n_harmonics
        self.grating_period = grating_period  # TODO add a y-grating and an x-grating
        self.wavelength = wavelength
        self.theta = theta
        # This remains constant throughout the entire computation
        self.kx0 = np.sin(self.theta)

        kx0 = np.zeros((2 * self.n_harmonics + 1), dtype=complex)
        kx0[self.n_harmonics] = self.kx0
        kx0 += np.array([2 * np.pi / self.grating_period *
                        n for n in range(-self.n_harmonics, self.n_harmonics + 1)])
        kx0 = np.diag(kx0)
        self.external_em_transfer_matrix = self.pq_matrices(self.top_layer, kx0)[
            1] / np.cos(self.theta) # This has a bug with off-angle incidence
        self.external_me_transfer_matrix = np.linalg.inv(
            self.external_em_transfer_matrix)  # TODO verify the other way as well

        self.top_layer.is_incident_layer = True

        # There should be no scattering matrix at the top. Rather, we should take the relative harmonics and multiply them to get the harmonics for every e, h x, y combination throughout the layers.

        for i, layer in enumerate(self.internal_layer_stack):
            kz_components, electric_field_harmonics, magnetic_field_harmonics = self.field_poynting_components(
                layer)
            # We want a set of coefficients so that these electric and magnetic field harmonics are consistent between layers
            forward_propagating_section = (electric_field_harmonics + \
                manual_matmul(self.external_me_transfer_matrix,
                              magnetic_field_harmonics, dtype = np.cdouble, threshold = 1e-11)) / 2
            backward_propagating_section = (electric_field_harmonics - \
                manual_matmul(self.external_me_transfer_matrix,
                              magnetic_field_harmonics, dtype = np.cdouble, threshold = 1e-11)) / 2

            phase_matrix = sp.linalg.expm(-layer.thickness * np.diag(kz_components)) # Must multiply by the angular frequency...in the z-direction?
            relation_matrix = np.vstack((forward_propagating_section, manual_matmul(backward_propagating_section, phase_matrix)))
            propagation_matrix = np.hstack((relation_matrix, np.flipud(relation_matrix)))
            dim = 2  * (2 * self.n_harmonics + 1)
            end_matrix = np.vstack((np.eye(dim), np.zeros((dim, dim))))
            backward_propagating_coefficients = np.linalg.solve(propagation_matrix, end_matrix)
            forward_propagating_coefficients = np.linalg.solve(propagation_matrix, np.flipud(end_matrix))
            print(f'Forward propagating coefficients:\n{forward_propagating_coefficients}')
            print(f'Backward propagating coefficients:\n{backward_propagating_coefficients}')
            should_be_zero = electric_field_harmonics @ phase_matrix @ forward_propagating_coefficients[:dim] + electric_field_harmonics @ forward_propagating_coefficients[dim:]
            print(f'Should be zero: {should_be_zero}')
        
        sys.exit(1)
        self.bottom_scattering_matrix = self.scattering_matrix(
            self.bottom_layer
        )

        self.internal_layer_scattering_matrices = [
            self.scattering_matrix(layer)
            for layer in self.internal_layer_stack
        ]

        self.global_scattering_matrix = self.top_scattering_matrix
        self.mode_coef_forward_preprop = []
        self.mode_coef_forward_postprop = []
        self.mode_coef_backward_preprop = []
        self.mode_coef_backward_postprop = []  # Python initializes by reference

        for i, layer_matrix in enumerate(self.internal_layer_scattering_matrices + [self.bottom_scattering_matrix]):
            self.global_scattering_matrix = extended_redheffer_star_product(
                self.global_scattering_matrix, layer_matrix)

        '''
        We compute the whole scattering matrix, and *then* the mode coefficients by moving the values around each matrix.
        '''

        print(
            f'Top (incident) scattering matrix:\n{self.top_scattering_matrix}')
        print(
            f'First internal scattering matrices:\n{self.internal_layer_scattering_matrices[0]}')
        print(f'Bottom scattering matrix:\n{self.bottom_scattering_matrix}')
        sys.exit(1)

    def field_poynting_components(self, layer: Layer_) -> np.ndarray:
        """
        Returns the Poynting vector components for a given layer.
        """
        kx = np.zeros((2 * self.n_harmonics + 1), dtype=complex)
        kx[self.n_harmonics] = self.kx0
        kx += np.array([2 * np.pi / self.grating_period *
                       n for n in range(-self.n_harmonics, self.n_harmonics + 1)])
        kx = np.diag(kx)
        p, q = self.pq_matrices(layer, kx)
        kz_components_squared, electric_field_harmonics = np.linalg.eig(
            manual_matmul(p, q))
        # Handle directionality here
        # Now we handle the case where we didn't have a FR matrix, and hence we have unbounded numbers of harmonics
        '''
        Check if the layer is marked as homogeneous. If not and the matrix is not full, raise an error saying that we might have resonance. If it is, check that all but two of the eigenvalues (the computed Poynting vectors in the z-direction) are zero; otherwise raise an error.
        '''
        kz_components = np.sqrt(kz_components_squared)
        # IF statement because PyTorch might take the correct or incorrect roots. Check that each one satisfies the conclusion to verify, since those correspond to the forward- or backward-propagating modes.
        # This must correspond to the direction of propagation, so if it's in the negative z-direction, we 
        # kz_components = np.where(np.imag(kz_components) < 0, np.conj(kz_components), kz_components)
        kz_components = np.where(np.real(kz_components) < 0, -np.conj(kz_components), kz_components)
        # Multiplication should be well defined here? TODO solve the other way and verify that the computed values are identical
        
        print(f'Kz components:\n{kz_components}')
        # print(f'Electric field harmonics:\n{electric_field_harmonics}')
        
        magnetic_field_harmonics = manual_matmul(manual_matmul(
            q, electric_field_harmonics), np.linalg.inv(np.diag(kz_components)))
        # magnetic_field_harmonics = manual_matmul(q, electric_field_harmonics)
        return kz_components, electric_field_harmonics, magnetic_field_harmonics

    def pq_matrices(self, layer: Layer_, kx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ky = np.diag(np.zeros((2 * self.n_harmonics + 1), dtype=complex) + np.array(
            [2 * np.pi / self.grating_period * n for n in range(-self.n_harmonics, self.n_harmonics + 1)]))
        permittivity_convolution, permeability_convolution = layer.layer_distribution_convolution_matrices()
        p00 = kx @ np.linalg.inv(permittivity_convolution) @ ky
        p01 = permeability_convolution - \
            kx @ np.linalg.inv(permittivity_convolution) @ kx
        p10 = ky @ np.linalg.inv(permittivity_convolution) @ ky - \
            permeability_convolution
        p11 = -ky @ np.linalg.inv(permittivity_convolution) @ kx
        p = np.block([[p00, p01], [p10, p11]])

        q00 = kx @ np.linalg.inv(permeability_convolution) @ ky
        q01 = permittivity_convolution - \
            kx @ np.linalg.inv(permeability_convolution) @ kx
        q10 = ky @ np.linalg.inv(permeability_convolution) @ ky - \
            permittivity_convolution
        q11 = -ky @ np.linalg.inv(permeability_convolution) @ kx
        q = np.block([[q00, q01], [q10, q11]])

        return p, q


    # def electric_field(self, layer: Layer_) -> np.ndarray:
    #     """
    #     Returns the electric field for a given layer.
    #     Does this by computing the mode coefficients for the system, but must be run AFTER the scattering matrices have been constructed. We will implement this later.
    #     """
    #     p, q = self.pq_matrices(layer, self.kx)
    #     omega_squared = np.sqrt(p @ q)
    #     eigenvalues, eigenvectors = np.linalg.eig(omega_squared)
    #     w, lambda_ = eigenvalues, np.sqrt(eigenvectors)
    #     return np.array([], dtype=np.cdouble)

