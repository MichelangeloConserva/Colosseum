from colosseum.emission_maps.base import StateLinear


class StateLinearOptimal(StateLinear):
    """
    The `StateLinearOptimal` emission map creates a non-tabular representation such that it is linear in the state value
    function of the optimal policy.
    """

    @property
    def V(self):
        return self._mdp.optimal_value_functions[1].ravel()
