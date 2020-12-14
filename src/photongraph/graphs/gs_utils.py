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
    qudits = state_vector.qudits

    assert state_vector.normalized, "State vector is not normalized."

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
                new_edge_qi = tuple(qudits[q] for q in new_edge)
                new_edges[new_edge] = bm_state_w
                edges[new_edge_qi] = bm_state_w

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


def qubit_stab_strings(stab_gens):
    """
    Generates a simple string representation for stabilizer generators
    of a 2-uniform qubit graph state.

    Each stabilizer generator must consist of X or X operations only

    Args:
        stab_gens (dict):

    Returns:
        (list): Contains all of the stabilizer generators as strings

    Todo: Rather than have this as a stand alone function it should be use to
          override the stab_gens method for teh QubitGraphState class.

    """
    qubit_num = len(stab_gens.keys())
    stab_strs = []
    # create a string of I's
    for q, stab_gen in stab_gens.items():
        stab_list = ['I'] * qubit_num
        ops = ['X', 'Z']
        for op, qs, w in stab_gen:
            if (op in ops) and len(qs) == 1 and w == 1:
                stab_list[qs[0]] = op
            else:
                raise Exception("Each operation of the stabilizer " +
                                "generator should be a tuple of the "
                                "form (label, qubits, weight) where "
                                "label is 'X' or 'Z' and len(qubits)==1"
                                "and weight ==1")

        stab_str = "".join(stab_list)
        stab_strs.append(stab_str)

    return stab_strs
