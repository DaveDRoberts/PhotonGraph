import numpy as np
from ..utils import basis_matrix


class StateVector:
    """
    Represents the state vector for a pure multi-qudit state in the
    computational basis. Amplitudes are stored in an array in the canonical
    order e.g. for 2 qubits we have (a_{00}, a_{01}, a_{10}, a_{11}).

    As default, Qudits are labelled in ascending order, from left to right,
    this is the opposite convention to qiskit.

    """

    def __init__(self, qudit_num, qudit_dim, vector=None, qudits=None, qudit_order_rev=False):
        """

        Args:
            qudit_num (int): Number of qudits >=1
            qudit_dim (int): Qudit dimension >=2
            vector (numpy.array): Amplitudes of computational basis states
            qudits (list[int]): Specifies qudit numbers
        """

        self._qudit_num = qudit_num
        self._qudit_dim = qudit_dim

        if not (vector is None):
            assert vector.shape == (qudit_dim**qudit_num,), \
                'Vector shape incompatible with qudit number and dimension.'
            self._vector = np.round(vector, 12).astype(np.complex128)
        else:
            self._vector = np.zeros(qudit_dim**qudit_num, dtype=np.complex128)

        if not (qudits is None):
            assert len(qudits) == qudit_num, \
                'Specified qudits is incompatible with qudit number.'
            self._qudits = sorted(set(qudits), reverse=qudit_order_rev)
        else:
            self._qudits = sorted(list(range(qudit_num)), reverse=qudit_order_rev)

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
            amp_mod = float(np.round(abs(amp), 8))
            amp_phase = float(np.round(np.angle(amp) / np.pi, 8))
            if not np.isclose(np.abs(amp), 0):
                basis_state_str = \
                    "|" + ''.join("%s " % ','.join(map(str, str(x)))
                                  for x in basis_state)[:-1] + ">"
                amp_str = f'{amp_mod:.8f} exp(i {amp_phase: .8f} pi) ' + "\n"
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
    def qudits(self):
        """ """
        return self._qudits

    @property
    def vector(self):
        """numpy.array: 1D Array to hold complex probability amplitudes"""
        return self._vector

    @property
    def normalized(self):
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

        self._vector = np.round(U @ self._vector, 12)

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

    def measure_Z(self, qudit, state=0):
        """
        Performs a computational basis measurement on a qudit.

        Args:
            qudit (int):
            state (int):

        Notes:
            reduced == True -> remove the measured qudit from the state
            if state == -1 then randomly project selected qudit on to one of the
            Z eigenstates.

        Todo: Write test for this funciton

        Todo: Implement random state measurement

        """

        d = self._qudit_dim
        n = self._qudit_num

        assert qudit in self._qudits, 'Invalid qudit.'
        assert state in list(range(-1, d))

        bm = basis_matrix(d, n)
        qudit_col = bm[:, self._qudits.index(qudit)]
        state_col = state*np.ones(qudit_col.shape[0], dtype=int).T
        m_bs_pos = np.argwhere(qudit_col - state_col == 0).T.flatten()

        new_vector = np.take(self._vector, m_bs_pos)

        self._vector = new_vector
        self.normalize()
        self._qudit_num -= 1
        self._qudits.remove(qudit)


