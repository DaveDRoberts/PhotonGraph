import pytest
from photongraph.graphs.edges import Edge


@pytest.mark.parametrize('edge, exp_result',
                         [(Edge((0, 1), 1, 2), 2),
                          (Edge((0, 1, 2), 2, 3), 3),
                          (Edge((2, 3, 4), 5, 7), 7)])
def test_qudit_dim(edge, exp_result):
    assert edge.qudit_dim == exp_result


@pytest.mark.parametrize('edge, exp_result',
                         [(Edge((0, 1), 1, 2), 1),
                          (Edge((0, 1, 2), 2, 3), 2),
                          (Edge((2, 3, 4), 5, 7), 5)])
def test_weight(edge, exp_result):
    assert edge.weight == exp_result


@pytest.mark.parametrize('edge, exp_result',
                         [(Edge((0, 1), 1, 2), (0, 1)),
                          (Edge((0, 1, 2), 2, 3), (0, 1, 2)),
                          (Edge((2, 3, 4), 5, 7), (2, 3, 4))])
def test_qudits(edge, exp_result):
    assert edge.qudits == frozenset(exp_result)


@pytest.mark.parametrize('edge, exp_result',
                         [(Edge((0, 1), 1, 2), 2),
                          (Edge((0, 1, 2), 2, 3), 3),
                          (Edge((2, 3, 4), 5, 7), 3)])
def test_cardinality(edge, exp_result):
    assert edge.cardinality == exp_result


@pytest.mark.parametrize('edge_1, edge_2, exp_result',
                         [(Edge((0, 1), 2, 3), Edge((0, 1), 1, 3), True),
                          (Edge((2, 3, 5), 2, 3), Edge((0, 1), 1, 3), False),
                          (Edge((2, 3, 5), 1, 2), Edge((2, 3, 5), 1, 2), True)])
def test_same_qudits(edge_1, edge_2, exp_result):
    assert edge_1.same_qudits(edge_2) == exp_result


@pytest.mark.parametrize('edge, m, exp_result',
                         [(Edge((0, 1), 2, 3), 4, 0),
                          (Edge((2, 3, 5), 2, 3), -1, 1),
                          (Edge((2, 3, 5), 1, 2), 1, 0)])
def test_add_weight(edge, m, exp_result):
    edge.add_weight(m)
    assert edge.weight == exp_result


@pytest.mark.parametrize('edge, m, exp_result',
                         [(Edge((0, 1), 2, 3), 4, 2),
                          (Edge((2, 3, 5), 2, 3), 2, 1),
                          (Edge((2, 3, 5), 1, 2), 1, 1)])
def test_mul_weight(edge, m, exp_result):
    edge.mul_weight(m)
    assert edge.weight == exp_result


@pytest.mark.parametrize('edge_1, edge_2, exp_result',
                         [(Edge((0, 1), 2, 3), Edge((0, 2), 1, 3),
                           Edge((0, 1, 2), 2, 3)),
                          (Edge((2, 3, 5), 2, 3), Edge((0, 1), 1, 3),
                           Edge((0, 1, 2, 3, 5), 2, 3)),
                          (Edge((2, 3, 5), 1, 2), Edge((2,3), 1, 2),
                           Edge((2, 3, 5), 1, 2))])
def test_edge_union(edge_1, edge_2, exp_result):
    assert edge_1.edge_union(edge_2) == exp_result
