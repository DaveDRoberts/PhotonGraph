from .open_graphstates import OpenGraphState


def flow(og_state):
    """
    Determines if an open graph state has a causal flow.

    Code developed from pseudocode algorithm found
    dx.doi.org/10.1007/978-3-540-70575-8_70 for finding a maximally delayed
    flow.

    Args:
        og_state (OpenGraphState):

    Returns:
        dict, dict, int

    Todo: Write some assert statements to validate input
    Todo: this function needs to fo into the OpenGraphState class

    """

    l = {}
    flow = {}
    #
    for v in og_state.outputs:
        l[v] = 0
    # Create sets for inputs, processed, and all vertices
    inputs = set(og_state.inputs)
    # processed = Out
    processed = set(og_state.outputs)
    # vertices = V
    vertices = set(og_state.qudits())
    correct = set(og_state.outputs).difference(inputs)

    # initialise k=1
    k = 1
    while True:

        correct_prime = set()
        processed_prime = set()

        for v in correct:
            ngh_v = set(og_state.neighbours(v))
            v_diff_o = vertices.difference(processed)
            u = ngh_v.intersection(v_diff_o)

            if len(u) == 1:
                flow[list(u)[0]] = v
                l[list(u)[0]] = k
                processed_prime.update(u)
                correct_prime.add(v)

        if not processed_prime:
            if processed == vertices:
                return l, flow, k
            return None

        else:

            processed.update(processed_prime)
            correct.difference_update(correct_prime)
            V_diff_In = vertices.difference(inputs)
            p_prime_inter = processed_prime.intersection(V_diff_In)
            correct.update(p_prime_inter)
            k += 1