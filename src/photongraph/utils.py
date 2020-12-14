import numpy as np
import itertools as it
import math


def check_integer(param, min_val=0):
    """Checks if param is a non-zero integer."""
    assert isinstance(param, int), "Must be an integer."
    assert param >= min_val, "Must be a greater than or equal to min_val."


def basis_matrix(qudit_dim, qudit_num):
    """
        Generates a matrix where the basis states are the rows in canonical
        order.

        Args:
            qudit_dim (int): Dimension of qudit.
            qudit_num (int): Number of qudits.

        Returns:
            numpy.array: A 2D array where every row is a basis state.

        Examples:
        >>> basis_matrix(2, 3)
        np.array([[0,0,0],
                  [0,0,1],
                  [0,1,0],
                  [0,1,1],
                  [1,0,0],
                  [1,0,1],
                  [1,1,0],
                  [1,1,1]])

        """
    n = qudit_num
    d = qudit_dim

    return np.array(list(it.product(*[list(range(d))] * n)))


def controlled_qubit_gates(qubit_num, U, U_label):
    """
    Generates all the possible n-qubit controlled-U gates for some single-qubit
    gate, U.

    Args:
        qubit_num:
        U: single-qubit unitary operation
        U_label (str): label for single-qubit operation e.g. X, Z

    Returns:
        (dict): Keys are gate names (str) and values are corresponding matrices
    """

    check_integer(qubit_num, 1)
    # Check that U is a 2x2 unitary matrix
    assert np.allclose(np.eye(2, dtype=complex), U.dot(U.T.conj()))
    # Check that U_label is a string
    assert type(U_label) == str

    I = np.array([[1, 0], [0, 1]], dtype=complex)
    P_zero = np.array([[1, 0], [0, 0]], dtype=complex)
    P_one = np.array([[0, 0], [0, 1]], dtype=complex)

    CU_gate_set = {}

    for m in range(2, qubit_num + 1):
        # m = gate qubit number
        m_qubit_combos = list(it.permutations(range(qubit_num), r=m))
        k = m - 1  # number of control qubits
        ctrls_targ = set(
            [(tuple(sorted(ct[:k])), ct[-1]) for ct in m_qubit_combos])

        # excludes the all-P_one combo
        ctrl_qubit_projs = list(it.product((P_zero, P_one), repeat=k))[:-1]

        # each combination of ctrls and targ specifies a control gate
        for ctrls, targ in ctrls_targ:
            CU_gate = np.zeros(shape=(2 ** qubit_num, 2 ** qubit_num))

            for cqp in ctrl_qubit_projs:
                temp = [I] * qubit_num
                for i, ctrl in enumerate(ctrls):
                    temp[ctrl] = cqp[i]
                CU_gate = np.add(CU_gate, tensor(*temp))

            temp2 = [I] * qubit_num
            for ctrl in ctrls:
                temp2[ctrl] = P_one
            temp2[targ] = U
            CU_gate = np.add(CU_gate, tensor(*temp2))

            ct = list(ctrls)
            ct.append(targ)
            CU_gate_name = "C" * k + U_label + "_" + "".join(
                np.array(ct, dtype=str))
            CU_gate_set[CU_gate_name] = CU_gate

    return CU_gate_set


def tensor(*matrices):
    """
    Function to calculate tensor product of multiple matrices.

    Args:
        matrices (numpy.ndarray): arbitary number of matrices

    Returns:
        (numpy.ndarray):

    Examples:
    >>>H = (1/np.sqrt(2)) * np.array([[1, 1],[1, -1]])
    >>>tensor(H,H)
    np.array([[ 0.5,  0.5,  0.5,  0.5],
              [ 0.5, -0.5,  0.5, -0.5],
              [ 0.5,  0.5, -0.5, -0.5],
              [ 0.5, -0.5, -0.5,  0.5]])

    """

    res = np.array([[1]])
    for i in matrices:
        res = np.kron(res, i)
    return res


def permute_matrix_rows(U, perm):
    """
    Permutes the rows of matrix U according to the a specified permutation
    The unpermuted matric has the rows in the order [0, 1, 2, 3, 4, 5, 6, 7],
    a permutation would be [0, 1, 6, 2, 7, 4, 3, 5]. So perm is a list where the
    number references the row of the matrix and its position in the list
    signifies its position.


    Args:
      U (numpy.ndarray): Matrix
      perm (list): permutation - length needs to be equal to the number of rows
                   and the numbers in that range must only appear once.

    Returns:
      (numpy.ndarray): Permutated matrix

    """

    U_copy = U.copy()
    og_order = list(range(len(perm)))

    U_copy[og_order] = U_copy[perm]

    return U_copy


def is_prime(n):
    """
    Checks if an integer n is prime.

    Code taken from:
    https://stackoverflow.com/questions/4114167/
    checking-if-a-number-is-a-prime-number-in-python

    Args:
        n (int):

    Returns:

    """
    check_integer(n)

    return n > 1 and all(n % i for i in it.islice(it.count(2),
                                                  int(math.sqrt(n)-1)))
