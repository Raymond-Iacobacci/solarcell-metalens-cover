import sys
import typing

import autograd.numpy as npa
import numpy as np
import numpy.typing as npt


# Everything in this file is set up for a single dimension of movement. It must be altered to allow for a second dimension.

def quar(mat: npt.ArrayLike) -> np.ndarray:
    assert mat.shape[0] == mat.shape[1]
    assert mat.shape[0] % 2 == 0
    mid = int(mat.shape[0] / 2)
    return np.array([mat[:mid, :mid], mat[:mid, mid:], mat[mid:, :mid], mat[mid:, mid:]])

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

def extended_redheffer_star_product(mat1: npt.ArrayLike, mat2: npt.ArrayLike, coef1: npt.ArrayLike, coef2: npt.ArrayLike) -> np.ndarray:
    '''
    Assume buffer between two systems
    '''
    s00, s01, s10, s11 = quar(mat1)
    

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
        # print(
        #     f'This is the permittivity matrix: {permittivity_convolution_matrix}')
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
        self.top_layer = layer_stack[0] # Semi-infinite
        self.bottom_layer = layer_stack[-1] # Semi-infinite
        self.n_harmonics = n_harmonics
        self.grating_period = grating_period
        self.wavelength = wavelength

        self.theta = theta
        self.k0 = 1  # Temporary for testing

        self.top_layer.is_incident_layer = True

        self.top_scattering_matrix = self.scattering_matrix(
            self.top_layer)
        
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
        
        for i, layer_matrix in enumerate(self.internal_layer_scattering_matrices):
            self.global_scattering_matrix, [self.mode_coef_forward_preprop, self.mode_coef_forward_postprop, self.mode_coef_backward_preprop, self.mode_coef_backward_postprop] = extended_
            
            
            '''
            Autocompleted via o1 preview
            '''
            # self.mode_coef_forward_preprop.append(np.zeros(
            #     (layer.shape[0], layer.shape[1]), dtype=np.cdouble
            # ))
            # self.mode_coef_forward_postprop.append(np.zeros(
            #     (layer.shape[0], layer.shape[1]), dtype=np.cdouble
            # ))
            # self.mode_coef_backward_preprop.append(np.zeros(
            #     (layer.shape[0], layer.shape[1]), dtype=np.cdouble
            # ))
            # self.mode_coef_backward_postprop.append(np.zeros(
            #     (layer.shape[0], layer.shape[1]), dtype=np.cdouble
            # ))
            # self.mode_coef_forward.append(np.zeros(
            #     (layer.shape[0], layer.shape[1]), dtype=np.cdouble
            # ))
            # self.mode_coef_backward.append(np.zeros(
            
            


        # These must have incident, internal, internal, transmission (what the fuck about the self-transmission? Find out what changes)
        self.mode_coef_forward = []
        self.mode_coef_backward = []
            
            
        self.transmission_scattering_matrix = self.scattering_matrix(
            self.transmission_layer
        )
        self.internal_layer_scattering_matrices = [
            self.scattering_matrix(layer) for layer in self.internal_layer_stack
        ]

        self.global_scattering_matrix = (
            self.incident_scattering_matrix
        )  # This can be arbitrarily set like jesus we are overcomplicating this
        # for i, internal_layer in enumerate(self.internal_layer_scattering_matrices):
        #     self.global_scattering_matrix, extra_coefs = self._RS_prod(
        #         self.global_scattering_matrix,
        #         internal_layer,
        #         [self.mode_coef_forward[i], self.mode_coef_backward[i]],
        #         [self.mode_coef_forward[i + 1], self.mode_coef_backward[i + 1]]
        #     )
        #     self.global_scattering_matrix = np.block([[self.global_scattering_matrix[0], self.global_scattering_matrix[2]], [self.global_scattering_matrix[1], self.global_scattering_matrix[3]]])
        
        self.global_scattering_matrix = self.incident_scattering_matrix
        
        
        
        
        for i, internal_layer_scattering_matrix in enumerate(self.internal_layer_scattering_matrices):
            self.global_scattering_matrix = redheffer_star_product(
                self.global_scattering_matrix,
                internal_layer_scattering_matrix
            )

        print(
            f'Incident scattering matrix:\n{self.incident_scattering_matrix}')

        self.global_scattering_matrix, extra_coefs = self._RS_prod(
            self.global_scattering_matrix,
            self.transmission_scattering_matrix,
            np.array([self.mode_coef_forward[-2],
                     self.mode_coef_backward[-2]]),
            np.array([self.mode_coef_forward[-1],
                     self.mode_coef_backward[-1]]),
        )
        self.global_scattering_matrix = np.block([[self.global_scattering_matrix[0], self.global_scattering_matrix[2]], [
                                                 self.global_scattering_matrix[1], self.global_scattering_matrix[3]]])

    def pq_matrices(
        self, layer: Layer_, kx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        ky shouldn't be a matrix, since we only use nondegenerate harmonics in the x-direction.
        """
        p00 = np.zeros(
            (self.n_harmonics * 2 + 1, self.n_harmonics * 2 + 1), dtype=np.cdouble
        )
        p01 = (
            layer.layer_distribution_convolution_matrices()[1]
            - kx
            @ np.linalg.inv(layer.layer_distribution_convolution_matrices()[0])
            @ kx
        )
        p10 = -layer.layer_distribution_convolution_matrices()[1]
        p11 = np.zeros(
            (self.n_harmonics * 2 + 1, self.n_harmonics * 2 + 1), dtype=np.cdouble
        )
        p = np.block([[p00, p01], [p10, p11]])
        # print(f"This is p for a layer with thickness {layer.thickness}:\n{p}")
        q00 = np.zeros(
            (self.n_harmonics * 2 + 1, self.n_harmonics * 2 + 1), dtype=np.cdouble
        )
        q01 = (
            layer.layer_distribution_convolution_matrices()[1]
            - kx
            @ np.linalg.inv(layer.layer_distribution_convolution_matrices()[1])
            @ kx
        )
        q10 = -layer.layer_distribution_convolution_matrices()[0]
        q11 = np.zeros(
            (self.n_harmonics * 2 + 1, self.n_harmonics * 2 + 1), dtype=np.cdouble
        )
        q = np.block([[q00, q01], [q10, q11]])
        # print(f"This is q for a layer with thickness {layer.thickness}:\n{q}")

        return p, q

    def scattering_matrix(self, layer: Layer_) -> np.ndarray:
        """
        Returns the scattering matrix for a given layer.
        """
        kx_norm = np.zeros((2 * self.n_harmonics + 1), dtype=np.cdouble)
        kx_norm[self.n_harmonics] = np.sqrt(
            self.incident_layer.permittivity[0]
            * self.incident_layer.permeability[
                0
            ]  # TODO This is a hack to get the correct value, but it should be calculated correctly.
            * np.sin(self.theta)
            * 2
            * np.pi
            / self.wavelength
        )  # Add in the sin function here, calculate vf the transformation matrix, use it to get the scattering matrices, we should be good...? Match them at the end
        kx_norm += [
            x * 2 * np.pi / self.grating_period
            for x in range(-self.n_harmonics, self.n_harmonics + 1)
        ]
        diag_kx_norm = np.diag(kx_norm)
        # print(f'This is the computed kx vector:\n{diag_kx_norm}')
        incident_diag_kz_norm = np.sqrt(
            self.incident_layer.permittivity[0] *
            self.incident_layer.permeability[0]
            - diag_kx_norm**2
        )  # May have to conjugate this, not really sure
        incident_diag_kz_norm = np.where(
            np.imag(incident_diag_kz_norm) < 0,
            np.conj(incident_diag_kz_norm),
            incident_diag_kz_norm,
        )
        print(f'Incident Kz in layer {layer.thickness}:\n{incident_diag_kz_norm}')
        p, q = self.pq_matrices(layer, diag_kx_norm)
        omega_squared = np.sqrt(
            p @ q
        )  # This intermediate step is useful for calculating the exact fields as they propagate.
        eigenvalues, eigenvectors = np.linalg.eig(
            omega_squared
        )  # This should have eigenvalues in the upper half plane. If not... \\ This is the solution to the field equations.
        # ... wait... do the HeSt simulation mathematics. There is an error when you approx Ky as 0 it seems like.
        # Now we calculate the A and B matrices via the W and V matrices. There are themselves calculated from the Eigenvectors of Omega. Let's see what they represent...
        kz_norm, e_lambda = np.sqrt(eigenvalues), (
            eigenvectors
        )  # Use sign conventions to handle these
        kz_norm = np.where(
            np.imag(kz_norm) < 0, -kz_norm, kz_norm
        )  # Check and then obfuscate
        diag_kz_norm = np.diag(kz_norm)
        print(
            f"Kz in layer {layer.thickness}\n{diag_kz_norm}")
        # Now we compute the wavevectors in different directions and spatial harmonics.

        h_lambda = (
            np.linalg.inv(p) @ e_lambda @ diag_kz_norm
        )  # Change so that diag is on the outside
        print(
            f"This is the h_lambda matrix for a layer of thickness: {layer.thickness}\n{h_lambda}")
        # Compute the global E to H transfer matrix
        e_to_m_transformer_matrix = np.hstack(  # This should be for the global case. Adjust this. Assume that the wave is incident on air, but make accomodations for the case that it is not.
            (
                np.vstack(
                    (
                        np.zeros(
                            (2 * self.n_harmonics + 1, 2 * self.n_harmonics + 1),
                            dtype=np.cdouble,
                        ),
                        incident_diag_kz_norm
                        + diag_kx_norm**2 @ np.linalg.inv(incident_diag_kz_norm),
                    )
                ),
                np.vstack(
                    (
                        -incident_diag_kz_norm,
                        np.zeros(
                            (2 * self.n_harmonics + 1, 2 * self.n_harmonics + 1),
                            dtype=np.cdouble,
                        ),
                    )
                ),
            )
        )
        # Compute the mode coupling coefficients
        # print(
        #     f"This is the e_to_m_transformer_matrix for a layer of thickness: {layer.thickness}\n{e_to_m_transformer_matrix}")
        preprop_a = e_lambda + \
            np.linalg.inv(e_to_m_transformer_matrix) @ h_lambda
        postprop_b = e_lambda - \
            np.linalg.inv(e_to_m_transformer_matrix) @ h_lambda
        # print(
        #     f"This is the preprop_a matrix for a layer of thickness: {layer.thickness}\n{preprop_a}")
        # print(
        #     f"This is the postprop_b matrix for a layer of thickness: {layer.thickness}\n{postprop_b}")
        prop_matrix = np.exp(
            -diag_kz_norm * self.k0 * layer.thickness
        )  # This k0 value should be calculated in the beginning as the magnitude of the Poynting vector
        # print(
        #     f"This is the prop_matrix for a layer of thickness: {layer.thickness}\n{prop_matrix}")
        preprop_b = postprop_b @ prop_matrix
        # print(
        #     f"This is the preprop_b matrix for a layer of thickness: {layer.thickness}\n{preprop_b}")
        mode_coefficient_generator = np.hstack(
            (np.vstack((preprop_a, preprop_b)), np.vstack((preprop_b, preprop_a)))
        )
        # print(
        #     f"This is the mode_coefficient_generator for a layer of thickness: {layer.thickness}\n{mode_coefficient_generator}")
        inv_mode_coefficient_generator = np.linalg.inv(
            mode_coefficient_generator)

        mode_coef_forward = inv_mode_coefficient_generator @ np.vstack(
            (
                np.eye(2 * (2 * self.n_harmonics + 1)),
                np.zeros(
                    (2 * (2 * self.n_harmonics + 1),
                     2 * (2 * self.n_harmonics + 1)),
                    dtype=np.cdouble,
                ),
            )
        )  # Obfuscate
        mode_coef_backward = inv_mode_coefficient_generator @ np.vstack(
            (
                np.zeros(
                    (2 * (2 * self.n_harmonics + 1),
                     2 * (2 * self.n_harmonics + 1)),
                    dtype=np.cdouble,
                ),
                np.eye(2 * (2 * self.n_harmonics + 1)),
            )
        )

        self.mode_coef_forward.append(mode_coef_forward)
        self.mode_coef_backward.append(mode_coef_backward)

        # Generation of these will produce an issue when run out of parallel. Enumerate them with a global lock to parallelize code.
        self.mode_coef_forward_preprop.append(
            mode_coef_forward[: 2 * (2 * self.n_harmonics + 1)]
        )
        self.mode_coef_forward_postprop.append(
            mode_coef_forward[2 * (2 * self.n_harmonics + 1):]
        )
        self.mode_coef_backward_preprop.append(
            mode_coef_backward[: 2 * (2 * self.n_harmonics + 1)]
        )
        self.mode_coef_backward_postprop.append(
            mode_coef_backward[2 * (2 * self.n_harmonics + 1):]
        )

        s00 = (
            e_lambda @ prop_matrix @ self.mode_coef_forward_preprop[-1]
            + e_lambda @ self.mode_coef_forward_postprop[-1]
        )
        print(
            f"This is the s00 matrix for a layer of thickness: {layer.thickness}\n{s00}")
        s01 = (
            e_lambda @ prop_matrix @ self.mode_coef_backward_preprop[-1]
            + e_lambda @ self.mode_coef_backward_postprop[-1]
            - np.eye((2 * (2 * self.n_harmonics + 1)), dtype=np.cdouble)
        )  # Check this, I think it should have the unity matrix scaled by the Poynting magnitude
        print(
            f"This is the s01 matrix for a layer of thickness: {layer.thickness}\n{s01}")
        s10 = (
            e_lambda @ self.mode_coef_forward_preprop[-1]
            + e_lambda @ prop_matrix @ self.mode_coef_forward_postprop[-1]
            - np.eye((2 * (2 * self.n_harmonics + 1)), dtype=np.cdouble)
        )
        print(
            f"This is the s10 matrix for a layer of thickness: {layer.thickness}\n{s10}")
        s11 = (
            e_lambda @ self.mode_coef_backward_preprop[-1]
            + e_lambda @ prop_matrix @ self.mode_coef_backward_postprop[-1]
        )
        print(
            f"This is the s11 matrix for a layer of thickness: {layer.thickness}\n{s11}")
        s = np.block([[s00, s01], [s10, s11]])
        print(
            f"This is the s matrix for a layer of thickness: {layer.thickness}\n{s}")
        return s

    def _RS_prod(self, Sm, Sn, Cm, Cn):
        # S11 = S[0] / S21 = S[1] / S12 = S[2] / S22 = S[3]
        # Cf = C[0] / Cb = C[1]
        Sn = np.array(Sn)
        Sm = np.array(Sm)
        Sn = [
            Sn[: 2 * (2 * self.n_harmonics + 1), : 2 *
               (2 * self.n_harmonics + 1)],
            Sn[2 * (2 * self.n_harmonics + 1):,
               : 2 * (2 * self.n_harmonics + 1)],
            Sn[: 2 * (2 * self.n_harmonics + 1), 2 *
               (2 * self.n_harmonics + 1):],
            Sn[2 * (2 * self.n_harmonics + 1):, 2 *
               (2 * self.n_harmonics + 1):],
        ]

        Sm = [
            Sm[: 2 * (2 * self.n_harmonics + 1), : 2 *
               (2 * self.n_harmonics + 1)],
            Sm[2 * (2 * self.n_harmonics + 1):,
               : 2 * (2 * self.n_harmonics + 1)],
            Sm[: 2 * (2 * self.n_harmonics + 1), 2 *
               (2 * self.n_harmonics + 1):],
            Sm[2 * (2 * self.n_harmonics + 1):, 2 *
               (2 * self.n_harmonics + 1):],
        ]
        tmp1 = np.linalg.inv(
            np.eye(2 * (2 * self.n_harmonics + 1), dtype=complex)
            - np.matmul(Sm[2], Sn[1])
        )
        tmp2 = np.linalg.inv(
            np.eye(2 * (2 * self.n_harmonics + 1), dtype=complex)
            - np.matmul(Sn[1], Sm[2])
        )

        # Layer S-matrix
        S11 = np.matmul(Sn[0], np.matmul(tmp1, Sm[0]))
        S21 = Sm[1] + np.matmul(Sm[3], np.matmul(tmp2,
                                np.matmul(Sn[1], Sm[0])))
        S12 = Sn[2] + np.matmul(Sn[0], np.matmul(tmp1,
                                np.matmul(Sm[2], Sn[3])))
        S22 = np.matmul(Sm[3], np.matmul(tmp2, Sn[3]))

        # Mode coupling coefficients
        C = [[], []]
        for m in range(len(Cm[0])):
            C[0].append(
                Cm[0][m] + np.matmul(Cm[1][m],
                                     np.matmul(tmp2, np.matmul(Sn[1], Sm[0])))
            )
            C[1].append(np.matmul(Cm[1][m], np.matmul(tmp2, Sn[3])))

        for n in range(len(Cn[0])):
            C[0].append(np.matmul(Cn[0][n], np.matmul(tmp1, Sm[0])))
            C[1].append(
                Cn[1][n] + np.matmul(Cn[0][n],
                                     np.matmul(tmp1, np.matmul(Sm[2], Sn[3])))
            )

        return [S11, S21, S12, S22], C

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
