import pytest
from photongraph.graphs.graphstates import GraphState
from photongraph.states.statevector import StateVector


@pytest.fixture
def qubit_graph_state():
    """Returns a GraphState instance with """
    weighted_edge_dict = {}
    qudit_dim = 2

    return GraphState(weighted_edge_dict, qudit_dim)


@pytest.fixture
def qudit_graph_state():
    """Returns a GraphState"""
    weighted_edge_dict = {}
    qudit_dim = 3

    return GraphState(weighted_edge_dict, qudit_dim)


@pytest.fixture
def qubit_hypergraph_state():
    """Returns a GraphState"""
    weighted_edge_dict = {}
    qudit_dim = 2

    return GraphState(weighted_edge_dict, qudit_dim)


@pytest.fixture
def qudit_hypergraph_state():
    """Returns a GraphState"""
    weighted_edge_dict = {}
    qudit_dim = 5

    return GraphState(weighted_edge_dict, qudit_dim)


def test_qudit_dim():
    pass

def test_qudit_num():
    pass

def test_qudits():
    pass

def test_edges():
    pass

def test_incidence_dict():
    pass

def test_stab_gens():
    pass

def test_state_vector():
    pass

def test_graph_hash():
    pass

def test_add_edges():
    pass

def test_adjacency():
    pass

def test_EPC():
    pass

def test_pivot():
    pass

def test_EM():
    pass

def test_ctrl_perm():
    pass

def test_measure_X():
    pass

def test_measure_Y():
    pass

def test_measure_Z():
    pass

def test_fusion():
    pass


# @pytest.mark.parametrize("d, n, state_vector, exp_result", [
#     (3, 3, state_vector_n3_d3_with_gp, {(1, 2): 2, (0, 1): 1, (0, 1, 2): 2}),
#     (3, 3, state_vector_n3_d3_ru_not_graph, {}),
#     (4, 4, state_vector_ghz_n4_d4, {}),
#     (4, 3, state_vector_n3_d4, {(1, 2): 2, (0, 1): 1, (0, 1, 2): 3}),
#     (2, 4, state_vector_n4_d2, {(2, 3): 1, (1, 3): 1, (1, 2): 1, (0, 1): 1})])
# def test_graph_state_edges(d, n, state_vector, exp_result):
#     assert graph_state_edges(d, n, state_vector) == exp_result
#
#
# @pytest.mark.parametrize("d, n, edges, exp_result", [
#     (4, 3, {(1, 2): 2, (0, 1): 1, (0, 1, 2): 3}, state_vector_n3_d4),
#     (2, 4, {(2, 3): 1, (1, 3): 1, (1, 2): 1, (0, 1): 1}, state_vector_n4_d2)])
# def test_state_vector_from_edges(d, n, edges, exp_result):
#     """
#     Checking if the produced dictionary is correct requires checking
#     each item in turn
#
#     """
#     test_result = state_vector_from_edges(d, n, edges)
#
#     assert np.all([np.isclose(test_result[state], exp_amp)
#                    for state, exp_amp in exp_result.items()])