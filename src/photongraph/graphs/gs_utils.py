import numpy as np
import itertools as it
from collections import defaultdict

from .graphstates import GraphState
from ..states.statevector import StateVector
from ..utils import basis_matrix


def state_check(state_vector, check_type):
    """
    Checks if state vector is is LME or RU

    Args:
        state_vector (StateVector):
        check_type (str):

    Returns:
        bool: Check result

    """

    assert check_type in ('LME', 'RU')

    n = state_vector.qudit_num
    d = state_vector.qudit_dim

    v = np.copy(state_vector.vector)
    amp_zero = v[0]
    amp_zero_tilda = np.round(1 / amp_zero, 10)
    v = amp_zero_tilda * v

    phis = np.angle(v)
    equal_sup_check = np.all(np.isclose(np.abs(v), np.ones(d ** n)))
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


def gs_from_sv(state_vector):
    """
    Generates a graph state from a state vector. If state vector does not
    correspond to a graph state an assertion error is raised.

    Args:
        state_vector (StateVector):

    Returns:
        GraphState:

    Raises:
        AssertionError: If equal_sup_check == False
        AssertionError: If RU_check.any() == True
        AssertionError: If state vector is not a graph state

    """

    n = state_vector.qudit_num
    d = state_vector.qudit_dim
    qudits = list(range(n))

    v = np.copy(state_vector.vector)
    amp_zero = v[0]
    amp_zero_tilda = np.round(1 / amp_zero, 10)
    v = amp_zero_tilda * v

    phis = np.angle(v)
    equal_sup_check = np.all(np.isclose(np.abs(v), np.ones(d ** n)))
    weights_um = np.array(np.round(phis * (d / (2 * np.pi)), 6))

    RU_check, weights = np.modf(np.mod(weights_um, d))

    # if RU_check.any() or (not equal_sup_check):
    #     assert 1 == 0, "State vector is NOT a graph state."

    assert not RU_check.any(), "State vector is NOT a RU state."
    assert equal_sup_check, "State vector is NOT an equal superposition " \
                            "of all basis states"

    weights = weights.astype(int)

    basis_mat = basis_matrix(d, n)

    state_vector_w = {tuple(bs): weight for bs, weight in
                      zip(basis_mat, weights)}

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
                gZ = edge_weight * np.prod(basis_mat[:, edge],
                                           axis=1).flatten()
                weights = np.mod(np.subtract(weights, gZ), d)

            state_vector_w = {tuple(bs): weight for bs, weight in
                              zip(basis_mat, weights)}

        if np.array(list(weights)).sum() == 0:
            break
        else:
            k = k + 1

        if k > n:
            assert 1 == 0, "State vector is NOT a graph state."

    return GraphState(edges, d, qudits)


def graph_state_edges(qudit_dim, qudit_num, state_vector):
    """
    For a given qudit state vector the weighted edges of the graph state
    are generated if it is a graph state.

    First, check if the all basis states are present in the state vector.
    Then check if the modulus of all amplitudes are equal, and if complex
    phases are all integer multiples of w=j2*pi/d. These are to check if
    the state is a roots of unity (RU) state (for qubits this is a real
    equally weighted (REW) state).

    There is a one-to-one correspondence between hypergraph states and REW
    states. However, not all RU states are qudit graph states.

    Args:
        qudit_dim (int): Qudit dimension
        qudit_num (int): Number of qudits
        state_vector (dict): Each key (tuple) is a basis state and values are
        the associated amplitudes.

    Returns:
        dict: Each value is an edge (tuple) and the values are weight of
        the edge. "Normal" qubit edges are always weight 1.

    """

    # generates a matrix where each row is a basis state
    basis_matrix = np.array(
        list(it.product(*[list(range(qudit_dim))] * qudit_num)))

    if not len(state_vector) == qudit_dim ** qudit_num:
        return {}
    # This removes any global phase by dividing all basis
    # state amps by the amp of the all-zero state
    amp_zero = np.round(state_vector[tuple(basis_matrix[0])], 8)
    if not amp_zero:
        return {}
    amp_zero_tilda = np.round(1 / amp_zero, 8)

    all_amps_normed = np.array(
        [state_vector[tuple(bs)] * amp_zero_tilda for bs in
         basis_matrix])
    phis = np.angle(all_amps_normed)

    weights_um = np.array(np.round(phis * (qudit_dim / (2 * np.pi)), 6))

    # check if state is a RU state, if not return empty {}
    RU_check, weights = np.modf(np.mod(weights_um, qudit_dim))
    equal_sup = np.all(np.isclose(np.abs(all_amps_normed),
                                  np.ones(qudit_dim ** qudit_num)))
    if RU_check.any() or (not equal_sup):
        return {}

    weights = weights.astype(int)

    state_vector_w = {tuple(bs): weight for bs, weight in
                      zip(basis_matrix, weights)}

    bm_states = list(it.product((0, 1), repeat=qudit_num))[1:]
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
                weights = np.mod(np.subtract(weights, gZ), qudit_dim)

            state_vector_w = {tuple(bs): weight for bs, weight in
                              zip(basis_matrix, weights)}

        # check if all signs are 0 i.e. + except for the |0>^tensor(n) state
        if np.array(list(weights)).sum() == 0:
            break
        else:
            k = k + 1
        # this means that the RU state is not a graph state
        if k > qudit_num:
            return {}

    return edges


def state_vector_from_edges(qudit_dim, qudit_num, edges):
    """
    Takes the edges of a graph state and generates the associated state vector.


    Args:
        qudit_dim (int): Qudit dimension
        qudit_num (int): Number of qudits
        edges (dict): Edges of graph state

    Returns:
        dict: state vector
    """

    # generates a matrix where each row is a basis state
    basis_matrix = np.array(
        list(it.product(*[list(range(qudit_dim))] * qudit_num)))

    new_weights = np.zeros(qudit_dim ** qudit_num, dtype=int)

    for edge, weight in edges.items():
        gZ = weight * np.prod(basis_matrix[:, edge], axis=1).flatten()
        new_weights = np.mod(np.add(new_weights, gZ), qudit_dim)

    state_vector_w = {tuple(bs): new_weight for bs, new_weight in
                      zip(basis_matrix, new_weights)}

    # convert from weight form to standard form
    amp = 1 / (np.sqrt(qudit_dim) ** qudit_num)
    state_vector = {
        state: np.round(amp * np.exp(2j * np.pi * weight / qudit_dim),
                        8) for
        state, weight in state_vector_w.items()}

    return state_vector
