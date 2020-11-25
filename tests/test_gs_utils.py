import pytest
from photongraph.graphs.gs_utils import *
from photongraph.states.statevector import StateVector


@pytest.fixture
def state_vector_n3_d3_with_gp():
    """Returns: A three qutrit graph state with a global phase."""

    n = 3
    d = 3
    vector = np.array(
        [0.1360827635 + 0.1360827635j,
         0.1360827635 + 0.1360827635j,
         0.1360827635 + 0.1360827635j,
         0.1360827635 + 0.1360827635j,
         0.0498097488 - 0.1858925105j,
         -0.1858925105 + 0.0498097488j,
         0.1360827635 + 0.1360827635j,
         -0.1858925105 + 0.0498097488j,
         0.0498097488 - 0.1858925105j,
         0.1360827635 + 0.1360827635j,
         0.1360827635 + 0.1360827635j,
         0.1360827635 + 0.1360827635j,
         -0.1858925105 + 0.0498097488j,
         0.0498097488 - 0.1858925105j,
         0.1360827635 + 0.1360827635j,
         0.0498097488 - 0.1858925105j,
         -0.1858925105 + 0.0498097488j,
         0.1360827635 + 0.1360827635j,
         0.1360827635 + 0.1360827635j,
         0.1360827635 + 0.1360827635j,
         0.1360827635 + 0.1360827635j,
         0.0498097488 - 0.1858925105j,
         0.0498097488 - 0.1858925105j,
         0.0498097488 - 0.1858925105j,
         -0.1858925105 + 0.0498097488j,
         -0.1858925105 + 0.0498097488j,
         -0.1858925105 + 0.0498097488j], dtype="complex128")

    return StateVector(n, d, vector)


@pytest.fixture
def state_vector_n3_d3_ru_not_gs():
    """Returns: A three qutrit state which is RU but not a graph state."""

    n = 3
    d = 3
    vector = np.array([0.1360827635 + 0.1360827635j,
                       0.0498097488 - 0.1858925105j,
                       0.1360827635 + 0.1360827635j,
                       -0.1858925105 + 0.0498097488j,
                       0.0498097488 - 0.1858925105j,
                       -0.1858925105 + 0.0498097488j,
                       0.0498097488 - 0.1858925105j,
                       -0.1858925105 + 0.0498097488j,
                       0.1360827635 + 0.1360827635j,
                       0.1360827635 + 0.1360827635j,
                       0.0498097488 - 0.1858925105j,
                       0.1360827635 + 0.1360827635j,
                       0.0498097488 - 0.1858925105j,
                       -0.1858925105 + 0.0498097488j,
                       0.1360827635 + 0.1360827635j,
                       0.1360827635 + 0.1360827635j,
                       0.1360827635 + 0.1360827635j,
                       -0.1858925105 + 0.0498097488j,
                       0.0498097488 - 0.1858925105j,
                       0.0498097488 - 0.1858925105j,
                       -0.1858925105 + 0.0498097488j,
                       0.0498097488 - 0.1858925105j,
                       0.0498097488 - 0.1858925105j,
                       0.1360827635 + 0.1360827635j,
                       0.1360827635 + 0.1360827635j,
                       0.1360827635 + 0.1360827635j,
                       0.1360827635 + 0.1360827635j], dtype="complex128")

    return StateVector(n, d, vector)


state_vector_n3_d3_lme_not_ru = {
    (0, 0, 0): 0.0272420087 - 0.1905122306j,
    (0, 0, 1): -0.1620693966 + 0.1037812494j,
    (0, 0, 2): -0.1699338368 - 0.0903301066j,
    (0, 1, 0): 0.059781768 - 0.1829294325j,
    (0, 1, 1): 0.1741579686 - 0.0818904057j,
    (0, 1, 2): -0.1640399455 + 0.1006376313j,
    (0, 2, 0): -0.17501983 + 0.0800318447j,
    (0, 2, 1): 0.0745483532 + 0.1774248562j,
    (0, 2, 2): 0.1776903374 - 0.0739133314j,
    (1, 0, 0): -0.1896129594 + 0.0329235884j,
    (1, 0, 1): -0.054484896 + 0.1845763613j,
    (1, 0, 2): -0.1043411236 + 0.1617095142j,
    (1, 1, 0): 0.1864516972 - 0.0476739069j,
    (1, 1, 1): -0.1377470985 - 0.134397815j,
    (1, 1, 2): 0.1083968756 + 0.1590193485j,
    (1, 2, 0): 0.1540507749 - 0.1153490163j,
    (1, 2, 1): -0.1812145578 - 0.0647944432j,
    (1, 2, 2): -0.1892200327 - 0.0351114662j,
    (2, 0, 0): 0.1012658226 + 0.1636528955j,
    (2, 0, 1): 0.0699690772 + 0.1792801307j,
    (2, 0, 2): 0.1036001369 + 0.1621852295j,
    (2, 1, 0): -0.1711345352 + 0.0880341278j,
    (2, 1, 1): -0.0265191346 + 0.190614195j,
    (2, 1, 2): -0.0285678862 + 0.1903179227j,
    (2, 2, 0): -0.156123493 + 0.1125277376j,
    (2, 2, 1): -0.0318508645 - 0.1897960965j,
    (2, 2, 2): -0.102339975 + 0.1629833284j}

state_vector_ghz_n4_d4 = {
    (0, 0, 0, 0): 1 / np.sqrt(2),
    (3, 3, 3, 3): 1 / np.sqrt(2)}

state_vector_n3_d4 = {
    (0, 0, 0): (0.125 + 0j),
    (0, 0, 1): (0.125 + 0j),
    (0, 0, 2): (0.125 + 0j),
    (0, 0, 3): (0.125 + 0j),
    (0, 1, 0): (0.125 + 0j),
    (0, 1, 1): (-0.125 + 0j),
    (0, 1, 2): (0.125 + 0j),
    (0, 1, 3): (-0.125 + 0j),
    (0, 2, 0): (0.125 + 0j),
    (0, 2, 1): (0.125 + 0j),
    (0, 2, 2): (0.125 + 0j),
    (0, 2, 3): (0.125 + 0j),
    (0, 3, 0): (0.125 + 0j),
    (0, 3, 1): (-0.125 + 0j),
    (0, 3, 2): (0.125 + 0j),
    (0, 3, 3): (-0.125 + 0j),
    (1, 0, 0): (0.125 + 0j),
    (1, 0, 1): (0.125 + 0j),
    (1, 0, 2): (0.125 + 0j),
    (1, 0, 3): (0.125 + 0j),
    (1, 1, 0): 0.125j,
    (1, 1, 1): (-0.125 + 0j),
    (1, 1, 2): -0.125j,
    (1, 1, 3): (0.125 + 0j),
    (1, 2, 0): (-0.125 + 0j),
    (1, 2, 1): (0.125 + 0j),
    (1, 2, 2): (-0.125 + 0j),
    (1, 2, 3): (0.125 + 0j),
    (1, 3, 0): -0.125j,
    (1, 3, 1): (-0.125 + 0j),
    (1, 3, 2): 0.125j,
    (1, 3, 3): (0.125 + 0j),
    (2, 0, 0): (0.125 + 0j),
    (2, 0, 1): (0.125 + 0j),
    (2, 0, 2): (0.125 + 0j),
    (2, 0, 3): (0.125 + 0j),
    (2, 1, 0): (-0.125 + 0j),
    (2, 1, 1): (-0.125 + 0j),
    (2, 1, 2): (-0.125 + 0j),
    (2, 1, 3): (-0.125 + 0j),
    (2, 2, 0): (0.125 + 0j),
    (2, 2, 1): (0.125 + 0j),
    (2, 2, 2): (0.125 + 0j),
    (2, 2, 3): (0.125 + 0j),
    (2, 3, 0): (-0.125 + 0j),
    (2, 3, 1): (-0.125 + 0j),
    (2, 3, 2): (-0.125 + 0j),
    (2, 3, 3): (-0.125 + 0j),
    (3, 0, 0): (0.125 + 0j),
    (3, 0, 1): (0.125 + 0j),
    (3, 0, 2): (0.125 + 0j),
    (3, 0, 3): (0.125 + 0j),
    (3, 1, 0): -0.125j,
    (3, 1, 1): (-0.125 + 0j),
    (3, 1, 2): 0.125j,
    (3, 1, 3): (0.125 + 0j),
    (3, 2, 0): (-0.125 + 0j),
    (3, 2, 1): (0.125 + 0j),
    (3, 2, 2): (-0.125 + 0j),
    (3, 2, 3): (0.125 + 0j),
    (3, 3, 0): 0.125j,
    (3, 3, 1): (-0.125 + 0j),
    (3, 3, 2): -0.125j,
    (3, 3, 3): (0.125 + 0j)}

state_vector_n4_d2 = {
    (0, 0, 0, 0): (0.25 + 0j),
    (0, 0, 0, 1): (0.25 + 0j),
    (0, 0, 1, 0): (0.25 + 0j),
    (0, 0, 1, 1): (-0.25 + 0j),
    (0, 1, 0, 0): (0.25 + 0j),
    (0, 1, 0, 1): (-0.25 + 0j),
    (0, 1, 1, 0): (-0.25 + 0j),
    (0, 1, 1, 1): (-0.25 + 0j),
    (1, 0, 0, 0): (0.25 + 0j),
    (1, 0, 0, 1): (0.25 + 0j),
    (1, 0, 1, 0): (0.25 + 0j),
    (1, 0, 1, 1): (-0.25 + 0j),
    (1, 1, 0, 0): (-0.25 + 0j),
    (1, 1, 0, 1): (0.25 + 0j),
    (1, 1, 1, 0): (0.25 + 0j),
    (1, 1, 1, 1): (0.25 + 0j)}


@pytest.mark.parametrize("d, n, state_vector, check_type, exp_result", [
    (3, 3, state_vector_n3_d3_with_gp, "LME", True),
    (3, 3, state_vector_n3_d3_with_gp, "RU", True),
    (3, 3, state_vector_n3_d3_ru_not_graph, "RU", True),
    (3, 3, state_vector_n3_d3_lme_not_ru, "RU", False),
    (3, 3, state_vector_n3_d3_lme_not_ru, "LME", True),
    (4, 4, state_vector_ghz_n4_d4, "RU", False)])
def test_state_check(d, n, state_vector, check_type, exp_result):
    assert state_check(d, n, state_vector, check_type) == exp_result


@pytest.mark.parametrize("d, n, state_vector, exp_result", [
    (3, 3, state_vector_n3_d3_with_gp, {(1, 2): 2, (0, 1): 1, (0, 1, 2): 2}),
    (3, 3, state_vector_n3_d3_ru_not_graph, {}),
    (4, 4, state_vector_ghz_n4_d4, {}),
    (4, 3, state_vector_n3_d4, {(1, 2): 2, (0, 1): 1, (0, 1, 2): 3}),
    (2, 4, state_vector_n4_d2, {(2, 3): 1, (1, 3): 1, (1, 2): 1, (0, 1): 1})])
def test_graph_state_edges(d, n, state_vector, exp_result):
    assert graph_state_edges(d, n, state_vector) == exp_result


@pytest.mark.parametrize("d, n, edges, exp_result", [
    (4, 3, {(1, 2): 2, (0, 1): 1, (0, 1, 2): 3}, state_vector_n3_d4),
    (2, 4, {(2, 3): 1, (1, 3): 1, (1, 2): 1, (0, 1): 1}, state_vector_n4_d2)])
def test_state_vector_from_edges(d, n, edges, exp_result):
    """
    Checking if the produced dictionary is correct requires checking
    each item in turn

    """
    test_result = state_vector_from_edges(d, n, edges)

    assert np.all([np.isclose(test_result[state], exp_amp)
                   for state, exp_amp in exp_result.items()])
