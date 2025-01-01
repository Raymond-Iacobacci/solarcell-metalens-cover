import sys
import typing
import scipy as sp  # This shouldn't have to be in here, it breaks minimal imports

import autograd.numpy as npa
import numpy as np
import numpy.typing as npt

logging = False
# Everything in this file is set up for a single dimension of movement. It must be altered to allow for a second dimension.


def quar(mat: npt.ArrayLike) -> np.ndarray:
    assert mat.shape[0] == mat.shape[1]
    assert mat.shape[0] % 2 == 0
    mid = int(mat.shape[0] / 2)
    return np.array([mat[:mid, :mid], mat[:mid, mid:], mat[mid:, :mid], mat[mid:, mid:]])


# TODO vectorize for GPU use and give errors
def manual_matmul(A, B, threshold=None, dtype=None):
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
    # This class has worked so far to elim half-zero errors
    result = np.zeros((a_rows, b_cols), dtype=dtype)

    # Perform the matrix multiplication manually
    for i in range(a_rows):
        for j in range(b_cols):
            for k in range(a_cols):  # or range(b_rows)
                result[i, j] += A[i, k] * B[k, j]
    if threshold is not None:
        result[np.abs(result) < threshold] *= 0
    return result


def extended_redheffer_star_product(mat1: npt.ArrayLike, mat2: npt.ArrayLike) -> np.ndarray:
    '''
    Assumes buffer between two systems.
    Compute the system scattering matrix and take the input from the bottom to be 0. Then use the reflected and the inputs to calculate backwards (while moving around the matrices) what the next coupling coefficients should be.
    '''
    sa00, sa01, sa10, sa11 = quar(mat1)
    sb00, sb01, sb10, sb11 = quar(mat2)
    identity = np.eye(sb00.shape[0])
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
        # Initialize convolution matrices
        size = 2 * self.n_harmonics + 1
        permittivity_convolution_matrix = np.zeros((size, size), dtype=np.cdouble)
        permeability_convolution_matrix = np.zeros((size, size), dtype=np.cdouble)

        # Compute Fourier representations with FFT
        permittivity_fourier = np.fft.fftshift(np.fft.fft(self.permittivity, axis=0))
        permeability_fourier = np.fft.fftshift(np.fft.fft(self.permeability, axis=0))

        # Extract relevant harmonics
        zero_harmonic = len(self.permittivity) // 2
        range_slice = slice(zero_harmonic - 2 * self.n_harmonics, zero_harmonic + 2 * self.n_harmonics + 1)
        permittivity_fourier = permittivity_fourier[range_slice] / len(permittivity_fourier)
        permeability_fourier = permeability_fourier[range_slice] / len(permeability_fourier)

        # Fill the convolution matrices using Toeplitz symmetry
        for x in range(size):
            for y in range(size):
                offset = y - x + 2 * self.n_harmonics
                permittivity_convolution_matrix[y, x] = permittivity_fourier[offset]
                permeability_convolution_matrix[y, x] = permeability_fourier[offset]

        return permittivity_convolution_matrix, permeability_convolution_matrix



def printdf(ipt: np.ndarray) -> None:
    from pandas import DataFrame
    np.set_printoptions(linewidth=120, precision=6, suppress=True)
    df = DataFrame(ipt)
    manual_format = "\n".join(
        " ".join(f"{val.real:+.6f}{val.imag:+.6f}j" for val in row) for row in ipt)
    print(df)


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
        self.external_em_transfer_matrix, self.external_me_transfer_matrix = self.pq_matrices(
            self.top_layer, kx0) / np.cos(self.theta)  # This has a bug with off-angle incidence
        # print(f'EM transfer matrix (identity?):\n{self.external_em_transfer_matrix}')
        # self.external_em_transfer_matrix = np.eye(self.external_em_transfer_matrix.shape[0])
        # self.external_me_transfer_matrix = np.eye(self.external_me_transfer_matrix.shape[0])

        if logging:
            print(
                f'External ME transfer matrix computed via PQ matrices:\n{self.external_me_transfer_matrix}')

        # self.external_me_transfer_matrix = np.linalg.inv(
        #     self.external_em_transfer_matrix)  # TODO verify the other way as well

        self.top_layer.is_incident_layer = True
        # Dimensionality fix
        dim = 2*(2 * self.n_harmonics + 1)
        a = np.eye(dim)
        b = np.zeros((dim, dim))
        self.global_scattering_matrix = np.block(
            [[b, a], [a, b]])  # Scattering matrix of free space

        # There should be no scattering matrix at the top. Rather, we should take the relative harmonics and multiply them to get the harmonics for every e, h x, y combination throughout the layers.

        self.W = []
        self.V = []

        # vacuum_interface_layer = Layer_(permittivity = np.ones(shape = self.top_layer.permittivity.shape), permeability = np.ones(shape = self.top_layer.permittivity.shape), thickness = 0, n_harmonics = 2*self.n_harmonics+1)
        kx = np.zeros((2 * self.n_harmonics + 1), dtype=complex)
        kx[self.n_harmonics] = self.kx0
        kx += np.array([2 * np.pi / self.grating_period *
                       n for n in range(-self.n_harmonics, self.n_harmonics + 1)])
        kx = np.diag(kx)
        print(f'Permittivity of top layer:\n{np.max(self.top_layer.permittivity)}')
        vac_p, vac_q = self.pq_matrices(self.top_layer, kx)
        print(f'Omega squared:')
        print(vac_p@vac_q)
        vac_lambda_w, vac_w = np.linalg.eig(vac_p@vac_q)
        print(
            f'Condition number of omega squared matrix:\n{np.linalg.cond(vac_p@vac_q, p = "fro")}')
        print(f'Determinant of omega squared:\n{np.linalg.det(vac_p@vac_q)}')
        print("Vacuum lambda (I have no idea if this is right or not at this point, I'm just copying the work from the middle layer's code)")
        print(vac_lambda_w)
        '''
        In this situation, the eigenvectors are the basis of the space R^6, ?all values independent?, and this one sits in front of the e^ solution.
        '''
        print(f'Vac lambda w before reordering\n{vac_lambda_w}')
        # TODO verify that the vacuum matrices work just like you verified that the layer matrices work
        # vac_lambda_w = np.sqrt(vac_lambda_w)
        vac_v = vac_q @ vac_w @ np.linalg.inv(np.diag(vac_lambda_w))
        # The bottom row should be negated, not sure why this works but if it's negated then the system breaks so...
        vac_trns_mat = np.block([[vac_w, vac_w], [-vac_v, vac_v]])
        print('Vacuum transfer matrix:')
        printdf(vac_trns_mat)
        '''
        Now the inverse method will potentially expose why this is an issue...perhaps when we switch domains the magnetic fields get inverted?
        '''

        print(f'Vacuum v:')
        printdf(vac_v)
        print(f'Vacuum w:')
        printdf(vac_w)
        
        
        

        for i, layer in enumerate(self.internal_layer_stack):
            print(f'Calculating scattering matrix for layer {i+2}')
            kx = np.zeros((2 * self.n_harmonics + 1), dtype=complex)
            kx[self.n_harmonics] = self.kx0
            kx += np.array([2 * np.pi / self.grating_period *
                           n for n in range(-self.n_harmonics, self.n_harmonics + 1)])
            kx = np.diag(kx)
            # good when increasing harmonics
            p, q = self.pq_matrices(layer, kx)
            '''
            NOTE
            Simulating across a single dimension creates incorrect decay of evanescent waves
            '''
            lambda_w, w = np.linalg.eig(p@q)
            print(f'Calculated lambda_w:\n{lambda_w}')
            # lambda_w = -1 * lambda_w # NOTE This line is necessary for single-harmonic work
            '''
            The following three lines have no effect on the propagation through a vacuum. Currently, it's reflecting everything when incident to a vacuum.
            '''

            # lambda_w = lambda_w[reorder]
            # w = w[:,reorder]
            print(f'Direct lambda_w:\n{lambda_w}')

            lambda_w = np.sqrt(lambda_w)
            # How should we handle this when we have two different versions of lambda_w that we use? Forward and backward?
            v = q @ w @ np.linalg.inv(np.diag(lambda_w))
            print(f'p')
            printdf(p)
            print(f'q')
            printdf(q)
            print(f'w')
            # why doesn't this have identity matrix as its [1,4:1,4] elements?
            printdf(w)
            print(f'v')
            printdf(v)
            # The identical results come from us setting up ky components. If we eliminate those...
            print(f'Lambda w (layer):\n{lambda_w}')
            '''
            Right now, evanescent modes going backwards (since we solve the equation for modes that *have* existed) are subject to immense gain when taking the inverse. We need to flip them, so invert those. It doesn't become singular but it gets close...
            We need to get one phase matrix going forwards while messing with the real parts, and one going backwards. Then deploy them individually inside the trans phase matrix.
            '''

            # fwd_lambda_w = -lambda_w #/ -1j
            fwd_lambda_w = lambda_w
            fwd_lambda_w = np.where(np.real(fwd_lambda_w) < 0, -fwd_lambda_w, fwd_lambda_w)
            fwd_phase_matrix = sp.linalg.expm(-layer.thickness * np.diag(fwd_lambda_w) * 2 * np.pi / self.wavelength)
            # TODO correct lambda forward to handle j
            fwd_lambda_w *= 1j
            bwd_lambda_w = lambda_w  # / -1j
            bwd_lambda_w = np.where(np.real(bwd_lambda_w) > 0, -bwd_lambda_w, bwd_lambda_w) #removed the = sign
            bwd_phase_matrix = sp.linalg.expm(layer.thickness * np.diag(bwd_lambda_w) * 2 * np.pi / self.wavelength)
            # TODO correct lambda backward to handle j
            bwd_lambda_w *= 1j

            print('Directional lambdas:')
            print(fwd_lambda_w)
            print(bwd_lambda_w)
            # TODO compare these two to ensure sign convention
            v_fwd = q @ w @ np.linalg.inv(np.diag(fwd_lambda_w))
            v_bwd = q @ w @ np.linalg.inv(np.diag(bwd_lambda_w))

            print('V matrices:')
            printdf(v_fwd)
            printdf(v_bwd)

            # we need to add imaginary coefficients to lambda. How do we determine which are forward propagating modes and which are backward propagating? This should be an element the single-harmonic as well...not sure why this doesn't make a difference...maybe no evanescent modes?
            phase_matrix = sp.linalg.expm(- layer.thickness *
                                          1j * np.diag(lambda_w) * 2 * np.pi/self.wavelength)

            print(np.linalg.det(phase_matrix))
            print('phase_matrix')

            printdf(phase_matrix)
            inv_phase_matrix = np.linalg.inv(phase_matrix)
            printdf(inv_phase_matrix)

            # NOTE sign convention extremely suspect here. need to verify
            trns_mat_phase = np.block(
                [[w @ bwd_phase_matrix, w @ fwd_phase_matrix], [v_bwd @ -bwd_phase_matrix, v_fwd @ fwd_phase_matrix]])
            trns_mat = np.block([[w, w], [-v_bwd, v_fwd]]) # removing = and adding negatives in front of these fixes it
            print('Determinants of directional phase matrices')
            print(np.linalg.det(fwd_phase_matrix))
            print(np.linalg.det(bwd_phase_matrix))
            # It looks like the calculations to get these are working fine. How do we find the solutions at the end? We should verify the signs of the 0-harmonic solutions in the phase matrix.
            print('Phase matrices')
            printdf(fwd_phase_matrix)
            printdf(bwd_phase_matrix)

            # trns_mat_phase = np.block([[w @ inv_phase_matrix, w @ phase_matrix], [v @ inv_phase_matrix, -v @ phase_matrix]])
            # trns_mat = np.block([[w, w], [v, -v]])
            print('Arrays part of three layer matrix calculation:')
            printdf(vac_trns_mat)
            # print(trns_mat_phase)
            print(f'Layer convolution matrices:')
            permit, permea = layer.layer_distribution_convolution_matrices()
            printdf(permit)
            printdf(permea)
            
            
            print('Trans matrix')
            # not good when increasing harmonics # becomes singular easily, like when introducing a single inhomogeneity
            printdf(trns_mat)
            print(np.linalg.det(trns_mat))
            print(np.linalg.inv(trns_mat))

            M = np.linalg.inv(trns_mat)@vac_trns_mat
            m11, m12, m21, m22 = quar(M)
            print('\n\n')
            # print(f'm11:\n{m11}')
            print('m11:')
            printdf(m11)
            # print(f'm12:\n{m12}')
            print('m12:')
            printdf(m12)
            # print(f'm21:\n{m21}')
            print('m21:')
            printdf(m21)
            # print(f'm22:\n{m22}')
            print('m22:')
            printdf(m22)
            print(np.linalg.inv(m22))
            print(-np.linalg.inv(m22)@m21)
            print(m12@np.linalg.inv(m22))
            # NOTE these values give the transmitted amplitudes. Compute the reflection and verify it with the Fresnel equations, and then compute the transmission with 1 - the square of that
            print(m12 @ np.linalg.inv(m22) @ m21)
            print('\n\n')

            # break

            three_layer_mat = np.linalg.inv(
                vac_trns_mat) @ trns_mat_phase @ np.linalg.inv(trns_mat) @ vac_trns_mat
            print(f'Master matrix:\n{np.abs(three_layer_mat)}')
            m11, m12, m21, m22 = quar(three_layer_mat)

            _id = np.zeros(shape=(2*self.n_harmonics + 1))
            _idi = np.zeros(shape=(2*self.n_harmonics + 1))
            _idi[self.n_harmonics] = 1
            id = np.hstack((_id, _idi))
            print(f'ID:\n{id}')
            id = np.linalg.inv(w) @ id
            print(f'ID 2:\n{id}')

            '''
            Not true--we must have 1-a=b since a and b destructively interfere at the boundaries.
            Also may not remain true when handling nonzero harmonics since light is at angles.
            '''
            # iden = np.eye(2 * self.n_harmonics + 1)
            # indicator = np.linalg.inv(m22) @ m21 @ id + np.linalg.inv(iden + m12) @ (iden - m11) @ id
            # print(f'Indicator (should be 0 everywhere):\n{indicator}')
            print(f'm11:\n{m11}')
            print(f'm12:\n{m12}')
            print(f'm21:\n{m21}')
            print(f'm22:\n{m22}')
            fref_coefs = -np.linalg.inv(m22) @ m21 @ id
            # print(f'{np.linalg.inv(m22)}')
            # print(f'{np.linalg.inv(m22) @ m21}')
            # print(f'Forward reflection:\n{fref_coefs}')
            ftrns_coefs = m11 @ id + m12 @ fref_coefs
            bref_coefs = m12 @ np.linalg.inv(m22) @ id
            btrns_coefs = np.linalg.inv(m22) @ id
            scattering_matrix = np.block(
                [[fref_coefs, ftrns_coefs], [btrns_coefs, bref_coefs]])
            print('\nScattering matrix:\n')
            printdf(scattering_matrix)
            print('\n')
            print(
                f'Forward transmission power:\n{np.sum(np.abs(ftrns_coefs)**2)}')
            print(
                f'Forward reflection power:\n{np.sum(np.abs(fref_coefs)**2)}')
            print(
                f'Backwards transmission power:\n{np.sum(np.abs(btrns_coefs)**2)}')
            print(
                f'Backwards reflection power:\n{np.sum(np.abs(bref_coefs)**2)}')
            self.global_scattering_matrix = extended_redheffer_star_product(self.global_scattering_matrix, scattering_matrix)
            '''
            Assert that lambda_w = lambda_v
            '''
            continue

            kz_components, electric_field_harmonics, magnetic_field_harmonics = self.field_poynting_components(
                layer)
            print(f'w the second way:\n{electric_field_harmonics}')
            # Must multiply by the angular frequency...in the z-direction?
            phase_matrix = sp.linalg.expm(-layer.thickness *
                                          np.diag(kz_components)*2*np.pi/self.wavelength)
            # magnetic_field_harmonics = np.block([[1, 0],[0, -1]])
            _, v_electric_field_harmonics, v_magnetic_field_harmonics = self.field_poynting_components(
                self.top_layer)  # Should be the newly constructed vacuum layer -- specific results like [[0, 1], [-1, 0]] should be q I think
            #
            # v_magnetic_field_harmonics = np.block([[-1, 0], [0, 1]])
            # magnetic_field_harmonics = np.block([[-1, 0], [0, 1]])
            '''
            This only refers to the phase matrix in the first vacuum layer. The second vacuum layer (since kx may not be the identity) is irrelevant since we solve for the fields at the interface; effectively we assume that the interface is of limiting thickness 0 in accordance with the Redheffer SP logic. Thus, there is no need for the vacuum phase matrix to handle calculations.
            '''
            # v_magnetic_field_harmonics *= -1
            # magnetic_field_harmonics *= -1
            v_transfer_matrix = np.block([[v_electric_field_harmonics, v_electric_field_harmonics], [
                                         -v_magnetic_field_harmonics, v_magnetic_field_harmonics]])
            # TODO compare v_transfer_matrix and v_transfer_matrix_2
            # print(v_transfer_matrix)

            print(f'Layer magnetic harmonics:\n{magnetic_field_harmonics}')
            magnetic_field_harmonics
            print(f'Layer electric harmonics:\n{electric_field_harmonics}')
            print(f'Vacuum transfer matrix:\n{v_transfer_matrix}')

            m_transfer_matrix_p = np.block([[electric_field_harmonics @ np.linalg.inv(phase_matrix), electric_field_harmonics @
                                           phase_matrix], [-magnetic_field_harmonics @ np.linalg.inv(phase_matrix), magnetic_field_harmonics @ phase_matrix]])
            m_transfer_matrix = np.block([[electric_field_harmonics, electric_field_harmonics], [
                                         -magnetic_field_harmonics, magnetic_field_harmonics]])

            print(
                f'Electric field harmonic solutions in the vacuum:\n{v_electric_field_harmonics}')
            print(
                f'Magnetic field harmonic solutions in the vacuum:\n{v_magnetic_field_harmonics}')

            '''
            This master matrix solves the equation for the downwards reflection and transmission coefficients, when a material layer is bookended by two vacuum layers. In effect, we solve the system of equations 
            1. Mv * [invpropv, propv] * coefsv = Mm * coefsm
            2. Mm * [invpropm, propm] * coefsm = Mv * coefsv
            to match the sx and sy components at both the interfaces, propagating the wave from its starting points. We can verify these results with standardized solvers for test cases, and we can back-compute the electric fields in the x- and y-directions to solve for the results. There IS an electric field in the z-direction since most waves will not be perfectly incident to the normal plane, but this is fixed via sx, sy, and p[oynting vector].
            '''
            print(
                f'v transfer matrix:\n{v_transfer_matrix}\nInverse:\n{np.linalg.inv(v_transfer_matrix)}\nDeterminant:\n{np.linalg.det(v_transfer_matrix)}')
            print(
                f'm transfer matrix:\n{m_transfer_matrix}\nInverse:\n{np.linalg.inv(m_transfer_matrix)}\nDeterminant:\n{np.linalg.det(m_transfer_matrix)}')

            master_matrix = np.linalg.inv(
                v_transfer_matrix) @ m_transfer_matrix_p @ np.linalg.inv(m_transfer_matrix) @ v_transfer_matrix

            # master_matrix = np.linalg.inv(v_transfer_matrix) @ np.linalg.inv(m_transfer_matrix) @ v_transfer_matrix

            print(f'Master matrix:\n{np.abs(master_matrix)}')
            m11, m12, m21, m22 = quar(master_matrix)

            _id = np.zeros(shape=(2*self.n_harmonics + 1))
            _idi = np.zeros(shape=(2*self.n_harmonics + 1))
            _idi[self.n_harmonics] = 1
            id = np.hstack((_idi, _id))
            print(f'ID:\n{id}')

            iden = np.eye(2 * self.n_harmonics + 1)
            # indicator = np.linalg.inv(m22) @ m21 @ id + np.linalg.inv(iden + m12) @ (iden - m11) @ id
            # print(f'Indicator (should be 0 everywhere):\n{indicator}')
            # print(m12+iden)
            print(f'm11:\n{m11}')

            print(f'm22:\n{m22}')
            fref_coefs = -np.linalg.inv(m22) @ m21 @ id
            print(f'{np.linalg.inv(m22)}')
            print(f'{np.linalg.inv(m22) @ m21}')
            print(f'Forward reflection:\n{fref_coefs}')
            ftrns_coefs = m11 @ id + m12 @ fref_coefs

            bref_coefs = m12 @ np.linalg.inv(m22) @ id
            btrns_coefs = np.linalg.inv(m22) @ id
            scattering_matrix = np.block(
                [[fref_coefs, ftrns_coefs], [btrns_coefs, bref_coefs]])
            # print(f'Layer {i} scattering matrix:\n{scattering_matrix}')
            print(f'Transmitted power ratio:\n{np.abs(ftrns_coefs)}')
            print(np.abs(fref_coefs))
            print(np.abs(btrns_coefs))
            print(np.abs(bref_coefs))

            '''
            We should have v_transfer_matrix @ transfer_matrix ** -1 @ v_transfer_matrix @ [1, a] = [b, 0]
            Then we have a = -m21/m22
            Then we have b = m11 - m12 * m21/m22
            We do one calculation top-down to get s00 and s01 and then another bottom-up to get the other halves of the scattering matrix.
            '''

            break
        self.ans = 1 - np.abs(self.global_scattering_matrix[0, 0])

    def soln(self):
        _id = np.zeros(shape=(2*self.n_harmonics + 1))
        _idi = np.zeros(shape=(2*self.n_harmonics + 1))
        _idi[0] = 1
        id = np.hstack((_idi, id))
        
        
        return self.global_scattering_matrix @ id

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
        kz_components = np.where(
            np.real(kz_components) < 0, -np.conj(kz_components), kz_components)
        if logging:
            print(f'Kz components:\n{kz_components}')
        # Multiplication should be well defined here? TODO solve the other way and verify that the computed values are identical

        magnetic_field_harmonics = manual_matmul(
            q, electric_field_harmonics, np.linalg.inv(np.diag(kz_components)))

        # magnetic_field_harmonics = manual_matmul(manual_matmul(
        #     q, electric_field_harmonics), np.linalg.inv(np.diag(kz_components*1j)))

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

    def zero_block(self) -> np.ndarray:
        return np.zeros(shape=(2 * self.n_harmonics + 1, 2 * self.n_harmonics + 1))

    def id_block(self) -> np.ndarray:
        return np.eye(2 * self.n_harmonics + 1)

    def pq_matrices_1d(self, layer: Layer_, kx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        e_conv_mat, u_conv_mat = layer.layer_distribution_convolution_matrices()

        p00 = self.zero_block()
        p01 = u_conv_mat - kx @ np.linalg.inv(e_conv_mat) @ kx
        p10 = -u_conv_mat
        p11 = self.zero_block()
        p = np.block([[p00, p01], [p10, p11]])

        q00 = self.zero_block()
        q01 = e_conv_mat - kx @ np.linalg.inv(u_conv_mat) @ kx
        q10 = -e_conv_mat
        q11 = self.zero_block()
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

# class Solver_v2: # v2
#     def __init__(
#         self, layer_stack: typing.List[Layer_],
        
#     )