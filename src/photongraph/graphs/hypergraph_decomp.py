import numpy as np
from collections import OrderedDict, defaultdict
import itertools
from ..utils import logical_basis

"""

The function decompose_REW() takes in some REW (real equally weighted) state 
and determines which Z operations (Z, CZ, CCZ etc. which are written as C1Z, 
C2Z, C3Z etc. so that there is pattern to the labelling) need to be applied to 
a n-qubit state with all qubits initialised in the |+> state. The code is 
inspired by the alogorithm described in https://arxiv.org/abs/1211.5554. 

The alogorithm is as follows:
    i) Take the REW state and find all the basis states which have single one 
       and a negative coefficient e.g. for n=3 -|010> 
   ii) Apply a Z (C1Z) to the qubit that is one in all those states e.g. Z_1 is 
       applied for -|010>, notice that although the minus sign is removed from 
       this state, minus signs appear on the other basis states where that 
       qubit has a value of one.
  iii) Next find all the states that have two qubits with the value one and a 
       minus and repeat the previous process but for C2Z gates. Notice that 
       fewer basis states are affected by this gate.
   iv) The process continues until all of the basis states have positive 
       cofficents. This can happen at any point but it must happen by the 
       application of CNZ where N is the total number of qubits i.e. a gate 
       which involves all of the qubits.

Notes about the code:

The REW state is stored in a dictionary where each key is a basis state and its value is its sign. The basis state is stored as
a tuple e.g. (1,0,0) for |100> and sign is saved as 0 for + and 1 for - i.e. (-1)^s for s = 0, we get +1 and for s=1 we get -1.

Qubits are ordered from left to right and indexed from 0 e.g. for |0,1,1,0>, qubit-0 = 0, qubit-1 = 1, qubit-2 = 1, qubit-3 = 0

"""


def decompose_REW(REW_state_OG, n, edgelist=False):
    """
    Takes in a REW state (format described above) determines the CkZ operations
    e.g. Z, CZ, CCZ etc. that must be applied to the n-qubit state
    |+>^tensor(n) to obtain the REW state. There is a one-to-one correspondence
    between REW states and hypergraph states. These CkZ operations correspond
    to the k-hyperedges of a hypergraph.

    Args:
        REW_state_OG (dict): The REW state is stored in a dictionary where each
                             key is a basis state and its value is its sign
        n (int): number of qubits
        edgelist (bool): If set to True, the function outputs a list of edges/
                         hyperedges. If False, a dictionary is returned.

    Returns:
        dict: keys are a string and values are a list of lists
              e.g. 'C2Z':[[0,1], [1,2]], 'C1Z':[[2],[0]]

        OR

        list: a list of lists where the inner lists are hyperedges
              e.g. [[0, 1], [1, 2]]

    Examples:

    TODO: I should create two functions, one chich returns CZ list and one
     which returns hyperedge list.
    """

    # make a copy of the input state, as it will be updated in the while loop
    REW_state = REW_state_OG.copy()
    # generates a matrix where each row is a binary bit string (they are in order)
    basis_matrix = np.array(list(itertools.product(*[(0, 1)] * n)))

    # basis manifolds are essentially the basis states grouped
    # by the number of ones they contain the key is the number
    # of ones and the value is a list of states
    basis_manifold = defaultdict(list)
    for i in REW_state.keys():
        basis_manifold[np.array(i).sum()].append(i)

    # k indexes the basis manifolds, start from 1 as the
    # sign of the all zero state is unaffected by Z operations
    k = 1

    # this stores the Z operations in the order Z, CZ, CCZ,
    Z_op_qubits_all = []

    while True:
        # generate a np array from the REW state signs
        old_signs = np.array(list(REW_state.values()), dtype=int)
        # this list holds the qubits which Z gates must be
        # applied to in order to remove minus sign
        Z_op_qubits = []
        # go through each state in the manifold
        for state in basis_manifold[k]:
            # checks if there is a minus sign in front of a basis state
            if REW_state[state] == 1:
                Z_op_qubits.append(
                    list(np.where(np.array(state) == 1)[0].flatten()))

                # if there are no corrections then signs remain the same
        if len(Z_op_qubits) == 0:
            new_signs = old_signs
        else:
            Z_op_qubits_all.append(Z_op_qubits)
            new_signs = gen_new_signs(Z_op_qubits, basis_matrix, old_signs)
            # update the REW state
            REW_state = {basis_state: new_signs[i] for i, basis_state in
                         enumerate(itertools.product(*[(0, 1)] * n))}

        # check if all signs are 0 i.e. + except for the |0>^tensor(n) state
        if np.array(list(new_signs))[1:].sum() == 0:
            break
        else:
            k = k + 1

    if edgelist:
        edges = []
        for ele in Z_op_qubits_all:
            for edge in ele:
                edges.append(edge)

        return edges

    else:
        # format Z operations into a dictionary
        Z_op_dict = {}
        for i, ele in enumerate(Z_op_qubits_all):
            Z_op_dict["C" + str(i + 1) + "Z"] = ele

        return Z_op_dict


def gen_new_signs(qubit_sets, basis_matrix, REW_signs):
    """
    All Z operations are diagonal in the computational basis. Therefore, is more efficient to
    represent them as vectors. Furthermore, since their values are either 1 or -1 and these values
    are just multiplied together it makes more sense to represent their values by 0 and 1 (+1 and -1).
    This means that multiplication of Z operations can be reduced to element-wise mod 2 addition.
    Another trick: the basis_matrix allows us to generate the vector for any CkZ operation just take
    columns of the matrix which correspond to the qubits involves in the CkZ gate and do element-wise
    multiplication. The result vector tells you which states the CkZ applies a minus sign.
    To generate the new signs, simply do addition mod 2 with the old signs vector and the the CkZ vector.


    Args:
        qubit_sets (list): contains one or more qubits
        basis_matrix (np.array): a matrix containing ordered binary bit strings as rows
        REW_signs (np.array): a vector containing 0s and 1s where 0 -> + and 1 -> -

    Returns:
        np.array: updated copy of the REW signs

    """
    REW_signs_new = REW_signs
    for qubit_set in qubit_sets:
        gZ = np.prod(basis_matrix[:, qubit_set], axis=1).flatten()
        REW_signs_new = np.mod(np.add(REW_signs_new, gZ), 2)
    return REW_signs_new


def qubit_REW_state_check(qubit_state, qubit_logical_basis=[], quiet=True):
    """
    This function checks if a qubit state is a REW state.

    Args:
        qubit_state (dict): keys = basis states, values = amplitudes
        qubit_logical_basis (list): A list of logical basis states in the canonical order.
        If this is not passed in, function will generate it. This gives flexibility for
        efficiency.
        quiet (bool): If True prints out informative check result

    Returns:
        (bool):Returns True if qubit state is a REW state

    Examples:

        >>>qubit_REW_state_check({(0,0):})

    """
    qubit_num = len(list(qubit_state.keys())[0])
    # if the logical basis isn't passed in calculate it
    # this is useful for checking many qubit states
    # since the logical basis doesn't need to be
    # calculatef each time.
    if not qubit_logical_basis:
        qubit_logical_basis = logical_basis(2, qubit_num)
    hilbert_space_size = 2 ** qubit_num

    # First check is if the number of terms in the n-qubit logical state is equal to 2**n
    if len(qubit_state) == hilbert_space_size:
        # extract amplitude for one of the basis states - this choice is arbitrary
        amp = np.round(qubit_state[tuple(list(qubit_state.keys())[0])], 8)
        # calculate reciprocal of amplitude
        amp_tilda = np.round(1 / amp, 8)
        # generate a vector with all values equal to amp_tilda
        amp_tilda_vec = np.full(shape=hilbert_space_size, fill_value=amp_tilda,
                                dtype=complex)
        # create an array to hold amplitudes of the basis states
        state_amps = np.zeros(hilbert_space_size, dtype=complex)
        for state, amp in qubit_state.items():
            state_amps[qubit_logical_basis.index(state)] = amp

        # element-wise multiplication
        norm_amps = np.round(amp_tilda_vec * np.array(state_amps))

        # check that all the basis state amplitudes are equal and real
        if np.all(np.abs(norm_amps) == 1.0) and np.allclose(norm_amps.imag,
                                                            np.zeros(len(
                                                                    norm_amps))):
            if not quiet:
                print("State is a REW qubit state.")
            return True
        else:
            if not quiet:
                print(
                    "All basis states are present but state is not a REW qubit state.")
            return False
    else:
        if not quiet:
            print("Some basis states are missing: not a REW qubit state")
        return False


def qubit_hyperedges(qubit_state, qubit_logical_basis=[]):
    """
    Function checks if logical qubit state is a REW state, if so returns a list of hyperedges.
    else returns an empty list.

    An n-qubit REW must be a superposition of all 2**n basis states where all of the basis
    state amplitudes are real and equal in magnitude. This function first checks if all the
    basis states have a non-zero amplitude since this is quickest method of ruling out a
    potential REW state. If this check is passed then the function checks if all amplitudes
    are REW. For this, take the reciprocal of one of the amplitudes and multiply all amplitudes
    by it. If the result of each multiplcation is 1 or -1 then the state is REW, otherwise it
    is not. After establishing if the state is REW, the hyperedges for the state are generated
    and returned.

    Note: There is a one-to-one correspondence between REW states and hypergraph states

    Args:
        qubit_state (dict): keys = basis states, values = amplitudes
        qubit_logical_basis (list): A list of logical basis states in the canonical order.
        If this is not passed in, function will generate it. This gives flexibility for efficiency.

    Returns:
        numpy.ndarray(): Array containing hyperedges only if state is REW, otherwise
        returns empty array

    Examples:
        >>>qubit_hyperedges()

        >>>qubit_hyperedges()

        >>>qubit_hyperedges()

    """

    if qubit_REW_state_check(qubit_state):
        qubit_num = len(list(qubit_state.keys())[0])

        qubit_logical_basis = logical_basis(2, qubit_num)
        if not qubit_logical_basis:
            qubit_logical_basis = logical_basis(2, qubit_num)
        hilbert_space_size = 2 ** qubit_num

        # create an array to hold amplitudes of the basis states
        state_amps = np.zeros(hilbert_space_size, dtype=complex)
        for state, amp in qubit_state.items():
            state_amps[qubit_logical_basis.index(state)] = amp
        # extract the signs of all the basis states: negative sign = 1 and positive sign = 0
        extract_signs = np.array((np.sign(state_amps) - 1) // -2, dtype=int)
        # makes the all-zero state positive, flip signs of all other basis states if negative
        if extract_signs[0] == 1: extract_signs = np.mod(extract_signs + 1, 2)
        # generates a dictionary to represent the logical qubit state
        REW_state_sign_form = {state: extract_signs[i] for i, state in
                               enumerate(qubit_logical_basis)}
        # generates a list of hyperedges, includes local Z operations
        hyper_graph_edges = np.array(
            decompose_REW(REW_state_sign_form, qubit_num, edgelist=True))
        return hyper_graph_edges
    else:
        return np.array([])


def state_vector_from_edges(qubit_num, edges):
    """
    Takes the edges of a graph state and generates the associated state vector.

    Each edge corresponds to the application of a CZ gate between the qubits
    which define the edge e.g. (2,3) <-> CZ_23.

    We start from the equal superposition state and apply the CZ gates.
    The application of a C^nZ gate results in multiplying the basis states
    where qubits of the C^nZ gate are 1.

    If using this to verify REW_decompose will need to take into account global phase

    Args:
        qubit_num (int): number of qubits in graph state
        edges (list): edges of graph state

    Returns:
        dict:
    """

    # initialise a dict for the equal-superposition state
    # just need a dict where the keys are the basis states and values are
    # 0 or 1.

    qubit_logical_basis = logical_basis(2, qubit_num)
    # bsf == binary sign form
    state_vector_bsf = {state: 0 for state in qubit_logical_basis}

    for edge in edges:
        for basis_state in qubit_logical_basis:
            if sum([basis_state[int(qb)] for qb in edge]) == len(edge):
                state_vector_bsf[basis_state] = \
                    np.mod((state_vector_bsf[basis_state] + 1), 2)

    # convert from bsf to standard form
    amp = 1/(np.sqrt(2)**qubit_num)
    state_vector_sf = {state: amp*(-1)**bin_sign for state, bin_sign in
                        state_vector_bsf.items()}

    return state_vector_sf
