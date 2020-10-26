import numpy as np
import itertools as it
from collections import defaultdict
from .graphs.graphstates import GraphState
from . import utils


class StateVector:
    """
    Represents the state vector for qudit state in the computational
    basis.

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
            self._vector = np.zeros(qudit_dim**qudit_num,
                                   dtype=np.complex128)

    def __repr__(self):
        n = self._qudit_num
        d = self._qudit_dim
        return f'StateVector(n = {n}, d = {d})'

    def __str__(self):

        state_str = ""
        for i, basis_state in enumerate(self._basis_matrix()):
            amp = self._vector[i]
            if not np.isclose(np.abs(amp), 0):
                basis_state_str = "|" + ''.join("%s " % ','.join(map(str, str(x))) for x in basis_state)[:-1] + ">"
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

    def _basis_matrix(self):
        """
        Generates a matrix where the basis states row.


        Returns:
            numpy.array:
        """
        n = self._qudit_num
        d = self._qudit_dim

        return np.array(list(it.product(*[list(range(d))] * n)))

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

        # create dict from basis matrix
        basis_matrix = self._basis_matrix()
        basis_dict = {tuple(bs): i for i, bs in enumerate(basis_matrix)}

        self._vector[basis_dict[tuple(basis_state)]] = amp

    def state_check(self, check_type):
        """
        Checks if state is LME, RU

        Returns:
            bool: check result

        """

        assert check_type in ('LME', 'RU')

        n = self._qudit_num
        d = self._qudit_dim

        v = np.copy(self._vector)
        amp_zero = v[0]
        amp_zero_tilda = np.round(1 / amp_zero, 10)
        v = amp_zero_tilda * v

        phis = np.angle(v)
        equal_sup_check = np.all(np.isclose(np.abs(v),np.ones(d ** n)))
        weights_um = np.array(np.round(phis * (d / (2 * np.pi)), 6))

        RU_check, weights = np.modf(np.mod(weights_um, d))

        if check_type == "LME":

            if equal_sup_check:
                return True
            else:
                return False

        elif check_type == "RU":
            if equal_sup_check and not (RU_check.any()):
                return True
            else:
                return False

    def _graph_state_edges(self):
        """
        This return s GraphState object if the state vector is a graph
        state.

        Returns:

        """

        # check if state vector is a RU state
        n = self._qudit_num
        d = self._qudit_dim

        v = np.copy(self._vector)
        amp_zero = v[0]
        amp_zero_tilda = np.round(1 / amp_zero, 10)
        v = amp_zero_tilda * v

        phis = np.angle(v)
        equal_sup_check = np.all(np.isclose(np.abs(v), np.ones(d ** n)))
        weights_um = np.array(np.round(phis * (d / (2 * np.pi)), 6))

        RU_check, weights = np.modf(np.mod(weights_um, d))

        if RU_check.any() or (not equal_sup_check):
            return {}

        weights = weights.astype(int)

        basis_matrix = self._basis_matrix()

        state_vector_w = {tuple(bs): weight for bs, weight in
                          zip(basis_matrix, weights)}

        bm_states = list(it.product((0, 1), repeat=n))[1:]
        basis_manifold = defaultdict(list)
        for bm_state in bm_states:
            basis_manifold[np.array(bm_state).sum()].append(bm_state)

        k = 1

        edges = {}

        while True:

            new_edges = {}

            for bm_state in basis_manifold[k]:
                bm_state_w = state_vector_w[bm_state]
                # checks if the basis state has a non-zero weight
                if bm_state_w:
                    new_edge = tuple(np.array(bm_state).nonzero()[0])
                    new_edges[new_edge] = bm_state_w
                    edges[new_edge] = bm_state_w

            if new_edges:
                for edge, edge_weight in new_edges.items():
                    gZ = edge_weight * np.prod(basis_matrix[:, edge],
                                               axis=1).flatten()
                    weights = np.mod(np.subtract(weights, gZ), d)

                state_vector_w = {tuple(bs): weight for bs, weight in
                                  zip(basis_matrix, weights)}

            if np.array(list(weights)).sum() == 0:
                break
            else:
                k = k + 1
            # this means that the RU state is not a graph state
            if k > n:
                return {}

        return edges

    @property
    def graph_state(self):
        """

        Returns:

        """
        d = self._qudit_dim
        n = self._qudit_num
        edges = self._graph_state_edges()
        qudits = list(range(n))

        assert edges, "State vector is NOT a graph state."

        return GraphState(edges, d, qudits)

    def logical_fock_states(self, d_enc, n_enc):
        """
        Generates the fock states which correspond to particular logical

        Args:
            d_enc (int): Qudit dimension encoding
            n_enc (int): Qudit number encoding

        Returns:
            np.ndarray
        """

        lfs = utils.logical_fock_states(d_enc, n_enc)
        qd_qb = utils.qudit_qubit_encoding(d_enc, n_enc)
        qb_qd = {v: k for k, v in qd_qb.items()}

        fock_states = []
        for bs in self._basis_matrix():
            fock_states.append(lfs[qb_qd[tuple(bs)]][0])

        return fock_states

    @property
    def dim(self):
        return self._qudit_dim

    @property
    def num(self):
        return self._qudit_num

    @property
    def vector(self):
        return self._vector




