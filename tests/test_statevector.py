import pytest
import numpy as np
from photongraph.states.statevector import StateVector
from photongraph.utils import intra_qubit_gate_set

vector_d4_n3 = np.array([(0.125 + 0j), (0.125 + 0j), (0.125 + 0j), (0.125 + 0j),
                         (0.125 + 0j), (-0.125 + 0j), (0.125 + 0j),
                         (-0.125 + 0j),
                         (0.125 + 0j), (0.125 + 0j), (0.125 + 0j), (0.125 + 0j),
                         (0.125 + 0j), (-0.125 + 0j), (0.125 + 0j),
                         (-0.125 + 0j),
                         (0.125 + 0j), (0.125 + 0j), (0.125 + 0j), (0.125 + 0j),
                         (0 + 0.125j), (-0.125 + 0j), (0 - 0.125j),
                         (0.125 + 0j),
                         (-0.125 + 0j), (0.125 + 0j), (-0.125 + 0j),
                         (0.125 + 0j),
                         (0 - 0.125j), (-0.125 + 0j), (0 + 0.125j),
                         (0.125 + 0j),
                         (0.125 + 0j), (0.125 + 0j), (0.125 + 0j), (0.125 + 0j),
                         (-0.125 + 0j), (-0.125 + 0j), (-0.125 + 0j),
                         (-0.125 + 0j),
                         (0.125 + 0j), (0.125 + 0j), (0.125 + 0j), (0.125 + 0j),
                         (-0.125 + 0j), (-0.125 + 0j), (-0.125 + 0j),
                         (-0.125 + 0j),
                         (0.125 + 0j), (0.125 + 0j), (0.125 + 0j), (0.125 + 0j),
                         (0 - 0.125j), (-0.125 + 0j), (0 + 0.125j),
                         (0.125 + 0j),
                         (-0.125 + 0j), (0.125 + 0j), (-0.125 + 0j),
                         (0.125 + 0j),
                         (0 + 0.125j), (-0.125 + 0j), (0 - 0.125j),
                         (0.125 + 0j)],
                        dtype='complex128')

vector_d2_n4 = 0.25 * np.ones(16, dtype='complex128')
vector_d2_n4_un = np.ones(16, dtype='complex128')

qgs_d2_n4 = intra_qubit_gate_set(16)
U_d2_n4 = qgs_d2_n4['CZ_01'] @ qgs_d2_n4['CZ_02'] @ qgs_d2_n4['CZ_03']
vector_d2_n4_evolved = np.array(
    [0.25 + 0.j, 0.25 + 0.j, 0.25 + 0.j, 0.25 + 0.j, 0.25 + 0.j, 0.25 + 0.j,
     0.25 + 0.j, 0.25 + 0.j, 0.25 + 0.j, -0.25 + 0.j, -0.25 + 0.j, 0.25 + 0.j,
     -0.25 + 0.j, 0.25 + 0.j, 0.25 + 0.j, -0.25 + 0.j])


@pytest.mark.parametrize('qudit_num, qudit_dim, vector', [
    (4, 2, vector_d2_n4),
    (3, 4, vector_d4_n3)
])
def test_qudit_dim(qudit_num, qudit_dim, vector):
    assert StateVector(qudit_num, qudit_dim, vector).qudit_dim == qudit_dim


@pytest.mark.parametrize('qudit_num, qudit_dim, vector', [
    (4, 2, vector_d2_n4),
    (3, 4, vector_d4_n3)
])
def test_qudit_num(qudit_num, qudit_dim, vector):
    assert StateVector(qudit_num, qudit_dim, vector).qudit_num == qudit_num


@pytest.mark.parametrize('qudit_num, qudit_dim, vector, exp_result', [
    (4, 2, vector_d2_n4, True),
    (4, 2, vector_d2_n4_un, False),
    (3, 4, vector_d4_n3, True)
])
def test_is_normalized(qudit_num, qudit_dim, vector, exp_result):
    assert StateVector(qudit_num, qudit_dim, vector).normalized == exp_result


@pytest.mark.parametrize('qudit_num, qudit_dim, vector', [
    (4, 2, vector_d2_n4_un)
])
def test_normalize(qudit_num, qudit_dim, vector):
    state_vector = StateVector(qudit_num, qudit_dim, vector)
    state_vector.normalize()
    assert state_vector.normalized


@pytest.mark.parametrize('qudit_num, qudit_dim, vector, unitary, exp_result', [
    (4, 2, vector_d2_n4, U_d2_n4, vector_d2_n4_evolved)
])
def test_evolve(qudit_num, qudit_dim, vector, unitary, exp_result):
    state_vector = StateVector(qudit_num, qudit_dim, vector)
    state_vector.evolve(unitary)
    assert np.array_equal(state_vector.vector, exp_result)


@pytest.mark.parametrize('qudit_num, qudit_dim, vector_a, vector_b, exp_result',
                         [(4, 2, vector_d2_n4, vector_d2_n4, 1.0 + 0.0j),
                          (4, 2, vector_d2_n4, vector_d2_n4_evolved, 0.5 + 0.0j)
])
def test_inner_product(qudit_num, qudit_dim, vector_a, vector_b, exp_result):
    state_vector_a = StateVector(qudit_num, qudit_dim, vector_a)
    state_vector_b = StateVector(qudit_num, qudit_dim, vector_b)

    assert state_vector_a.inner_product(state_vector_b) == exp_result


@pytest.mark.parametrize('qudit_num, qudit_dim, vector, basis_state, exp_result',
                         [(4, 2, vector_d2_n4, [0, 1, 1, 0], 0.25 + 0.0j),
                          (3, 4, vector_d4_n3, [3, 2, 2], -0.125 + 0.0j)
])
def test_get_amp(qudit_num, qudit_dim, vector, basis_state, exp_result):
    state_vector = StateVector(qudit_num, qudit_dim, vector)

    assert state_vector.get_amp(basis_state) == exp_result


@pytest.mark.parametrize('qudit_num, qudit_dim, vector, basis_state, new_amp',
                         [(4, 2, vector_d2_n4, [0, 1, 1, 0], 0.0 + 0.0j),
                          (3, 4, vector_d4_n3, [3, 2, 2], 0.5 + 0.0j)
])
def test_set_amp(qudit_num, qudit_dim, vector, basis_state, new_amp):
    state_vector = StateVector(qudit_num, qudit_dim, vector)
    state_vector.set_amp(basis_state, new_amp)

    assert state_vector.get_amp(basis_state) == new_amp








