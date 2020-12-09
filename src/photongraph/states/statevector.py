import numpy as np

from ..utils import logical_fock_states, basis_matrix, qudit_qubit_encoding


class StateVector:
    """
    Represents the state vector for a pure multi-qudit state in the
    computational basis. Amplitudes are stored in an array in the canonical
    order e.g. for 2 qubits we have (a_{00}, a_{01}, a_{10}, a_{11}).

    Attributes:
          _qudit_num (int):
          _qudit_dim (int):
          _vector (numpy.array):

    """

    def __init__(self, qudit_num, qudit_dim, vector=None):
        """

        Args:
            vector (numpy.array): Amplitudes of computational basis states
            qudit_num (int): Number of qudits >=1
            qudit_dim (int): Qudit dimension >=2
        """

        # check that the length of the vector is compatible with the
        # number of qudits and qudit dimension.

        # make sure the data type of the np array is complex

        self._qudit_num = qudit_num
        self._qudit_dim = qudit_dim

        if not (vector is None):
            self._vector = vector
        else:
            self._vector = np.zeros(qudit_dim**qudit_num, dtype=np.complex128)

    def __repr__(self):
        n = self._qudit_num
        d = self._qudit_dim
        return f'StateVector(n = {n}, d = {d})'

    def __str__(self):

        d = self._qudit_dim
        n = self._qudit_num
        state_str = ""
        for i, basis_state in enumerate(basis_matrix(d, n)):
            amp = self._vector[i]
            if not np.isclose(np.abs(amp), 0):
                basis_state_str = \
                    "|" + ''.join("%s " % ','.join(map(str, str(x)))
                                  for x in basis_state)[:-1] + ">"
                amp_str = str(amp) + "\n"
                state_str += basis_state_str + " : " + amp_str

        if state_str:
            return f'{state_str}'
        else:
            return 'Null Vector'

    def __eq__(self, other):
        assert isinstance(other, self.__class__)

        same_vector = np.allclose(self._vector, other.vector)
        same_qudit_dim = other.qudit_dim == self._qudit_dim
        same_qudit_num = other.qudit_num == self._qudit_num

        return same_vector and same_qudit_dim and same_qudit_num

    @property
    def qudit_dim(self):
        """int: Dimension of qudit."""
        return self._qudit_dim

    @property
    def qudit_num(self):
        """int: Number of qudits."""
        return self._qudit_num

    @property
    def vector(self):
        """numpy.array: 1D Array to hold complex probability amplitudes"""
        return self._vector

    @property
    def is_normalized(self):
        """bool: checks if state vector is normalised"""
        return np.isclose(np.sum(np.square(np.abs(self._vector))), 1.0)

    def evolve(self, U):
        """
        Applies a unitary matrix to the vector.

        Args:
            U (numpy.ndarray): Unitary matrix.

       Todo: Check that U is unitary and that dimensions are compatible with
             vector.

        """

        self._vector = U @ self._vector

    def inner_product(self, state):
        """
        Takes the inner product between itself and some other state.

        Args:
            state (StateVector): State vector.

        Returns:
            complex: A complex, scalar value.

        """

        assert isinstance(state, self.__class__)
        assert self._vector.shape == state.vector.shape

        return state.vector.T.conj() @ self._vector

    def normalize(self):
        """
        Normalizes the probability amplitudes of the state vector.

        """

        v = self._vector
        norm_const = np.sqrt(np.sum(np.square(np.abs(v))))
        self._vector = self._vector / norm_const

    def get_amp(self, basis_state):
        """
        Gets the amplitude of a specified computational basis state.

        Args:
            basis_state (list): Computational basis state.

        Returns:
            complex: Complex probability amplitude.

        Returns:

        """
        d = self._qudit_dim
        n = self._qudit_num
        # create dict from basis matrix
        basis_mat = basis_matrix(d, n)
        basis_dict = {tuple(bs): i for i, bs in enumerate(basis_mat)}

        return self._vector[basis_dict[tuple(basis_state)]]

    def set_amp(self, basis_state, amp):
        """
        Sets the amplitude of a specified computational basis state.

        Args:
            basis_state (list): Computational basis state.
            amp (complex): Complex probability amplitude.

        Returns:

        """
        d = self._qudit_dim
        n = self._qudit_num
        # create dict from basis matrix
        basis_mat = basis_matrix(d, n)
        basis_dict = {tuple(bs): i for i, bs in enumerate(basis_mat)}

        self._vector[basis_dict[tuple(basis_state)]] = amp

    def logical_fock_states(self, d_enc, n_enc):
        """
        Generates the fock states which correspond to particular logical states
        for a specified photon encoding.

        Args:
            d_enc (int): Qudit dimension encoding
            n_enc (int): Qudit number encoding

        Returns:
            np.ndarray
        """

        d = self._qudit_dim
        n = self._qudit_num

        lfs = logical_fock_states(d_enc, n_enc)
        qd_qb = qudit_qubit_encoding(d_enc, n_enc)
        qb_qd = {v: k for k, v in qd_qb.items()}

        fock_states = []
        for bs in basis_matrix(d, n):
            fock_states.append(lfs[qb_qd[tuple(bs)]][0])

        return fock_states