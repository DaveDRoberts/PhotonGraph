from collections import defaultdict
import itertools as it
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
            vector (numpy.array): Amplitudes of computational basis
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
        if np.allclose(self._vector, other.vector):
            return True
        else:
            return False

    @property
    def qudit_dim(self):
        return self._qudit_dim

    @property
    def qudit_num(self):
        return self._qudit_num

    @property
    def vector(self):
        return self._vector

    def evolve(self, U):
        """

        Check that dimensions of U are compatible with the vector.
        Check that U is unitary


        Args:
            U:

        Returns:

        """

        self._vector = U @ self._vector

    def inner_product(self, state):
        """

        Check that state is compatible  with state vector
        Check that state is type StateVector
        Args:
            state (numpy.array):

        Returns:

        """
        return state.T.conj() @ self._vector

    def normalize(self):
        """

        Returns:

        """

        v = self._vector
        norm_const = np.sqrt(np.sum(np.square(np.abs(v))))
        self._vector = self._vector / norm_const

    def schmidt_measure(self):
        """
        Computes the Schmidt measure for the state vector

        Returns:

        """
        return NotImplementedError

    def set_amp(self, basis_state, amp):
        """

        Args:
            basis_state (list):
            amp (complex)

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
        Generates the fock states which correspond to particular logical

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