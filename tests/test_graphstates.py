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


