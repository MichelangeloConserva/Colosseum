from typing import TYPE_CHECKING, Iterable, List, Tuple, Union

import numpy as np
from dm_env import TimeStep

if TYPE_CHECKING:
    from colosseum.mdps import ContinuousMDP, EpisodicMDP


def human_loop(mdp: Union["ContinuousMDP", "EpisodicMDP"], other_policies: dict = None):
    """
    allows a human to control an MDP.
    """
    verba = lambda: print(mdp.get_grid_repr())

    print("Start calculating the optimal policy")
    _ = mdp.optimal_policy
    print("End calculating the optimal policy")

    state = mdp.reset()
    while True:
        print("State:", state)
        verba()

        if mdp.is_episodic():
            p = np.argmax(
                mdp.optimal_policy().pi(mdp.h, mdp.node_to_index(mdp.cur_node))
            )
        else:
            p = np.argmax(mdp.optimal_policy().pi(mdp.node_to_index(mdp.cur_node)))
        print(f"The optimal action for this state is:{p}")

        if other_policies is not None:
            for pi_name, pi in other_policies.items():
                print(
                    f"The action of policy {pi_name} for this state is:{np.argmax(pi[mdp.cur_node])}"
                )

        action = int(
            input(
                "Available actions are: "
                + ",".join(map(str, range(mdp.action_spec().num_values)))
                + ".\tChoose one to act or type anything else to terminate.\n"
            )
        )
        if action not in range(mdp.action_spec().num_values):
            break
        state = mdp.step(action)
        if state.last():
            print("State:", state)
            state = mdp.reset()


def random_loop(
    mdp: Union["ContinuousMDP", "EpisodicMDP"],
    N: int,
    verbose: bool = False,
    return_actions: bool = False,
    human_readable=False,
) -> Union[Tuple[List[TimeStep], List[int]], List[TimeStep]]:
    """
    generates interactions data by selecting actions a random.

    Parameters
    ----------
    mdp: Union["ContinuousMDP", "EpisodicMDP"]
        the MDP instance.
    N : int
        the number of interactions.
    verbose : bool, optional
        checks whether to print verbose outputs. Be default, it is set to False.
    return_actions: bool, optional
        checks whether to return the action played. By default, it is set to False,
    human_readable: bool
        checks whether to print the state information in a human interpretable form. By default, it is set to False.
    """

    if human_readable:
        verba = lambda: print(mdp.get_grid_repr())
    else:
        verba = lambda: print("State:", state, "Action: ", action)

    states = []
    state = mdp.reset()
    states.append(state)
    actions = []
    action = None
    while len(states) < N:
        if verbose:
            verba()
        state, action = mdp.random_step()
        if return_actions:
            actions.append(action)
        states.append(state)
        if state.last():
            if verbose:
                print("Last state:", state)
            state = mdp.reset()
            states.append(state)

    if return_actions:
        return states, actions
    return states


def prefixed_action_loop(
    mdp: Union["ContinuousMDP", "EpisodicMDP"],
    actions: Iterable[int],
    verbose: bool = False,
    human_readable: bool = False,
) -> List[TimeStep]:
    """
    generates interaction with the MDP according to the actions given in input.

    Parameters
    ----------
    mdp: Union["ContinuousMDP", "EpisodicMDP"]
        the MDP instance.
    actions : Iterable[int]
        the action that are to be played.
    verbose : bool, optional
        checks whether to print verbose outputs. Be default, it is set to False.
    human_readable: bool
        checks whether to print the state information in a human interpretable form. By default, it is set to False.
    """

    if human_readable:
        verba = lambda: print(mdp.get_grid_repr())
    else:
        verba = lambda: print("State:", state, "Action: ", action)

    states = []
    state = mdp.reset()
    states.append(state)
    for action in actions:
        if verbose:
            verba()
        state = mdp.step(action)
        states.append(state)
        if state.last():
            if verbose:
                print("Last state:", state)
            state = mdp.reset()
            states.append(state)
    return states
