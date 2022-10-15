from typing import TYPE_CHECKING, Iterable, List, Tuple, Union

import numpy as np
from dm_env import TimeStep

from colosseum import config

if TYPE_CHECKING:
    from colosseum.mdp import ContinuousMDP, EpisodicMDP


def human_loop(mdp: Union["ContinuousMDP", "EpisodicMDP"], other_policies: dict = None):
    """
    allows a human to control an MDP.
    """

    verba = lambda: print(mdp.get_grid_representation(mdp.cur_node))

    print("Start calculating the optimal policy")
    optimal_policy = mdp.get_optimal_policy(False)
    print("End calculating the optimal policy")

    state = mdp.reset()
    while True:
        print("State:", state)
        verba()

        if mdp.is_episodic():
            optimal_action = optimal_policy[mdp.h, mdp.node_to_index[mdp.cur_node]]
        else:
            optimal_action = optimal_policy[mdp.node_to_index[mdp.cur_node]]
        print(f"The optimal action for this state is:{optimal_action}")

        if other_policies is not None:
            for pi_name, pi in other_policies.items():
                print(
                    f"The action of policy {pi_name} for this state is:{np.argmax(pi[mdp.cur_node])}"
                )

        action = int(
            input(
                "Available actions are: "
                + ",".join(map(str, range(mdp.n_actions)))
                + ".\tChoose one to act or type anything else to terminate.\n"
            )
        )
        if action not in range(mdp.n_actions):
            break
        state = mdp.step(action)
        if state.last():
            print("State:", state)
            state = mdp.reset()


def random_loop(
    mdp: Union["ContinuousMDP", "EpisodicMDP"],
    N: int,
    return_actions: bool = False,
    human_readable=False,
) -> Union[Tuple[List[TimeStep], List[int]], List[TimeStep]]:
    """
    generates interactions data by selecting actions a random.

    Parameters
    ----------
    mdp: Union["ContinuousMDP", "EpisodicMDP"]
        The MDP instance.
    N : int
        The number of interactions.
    return_actions: bool, optional
        If True, the selected actions are returned. By default, it is set to False,
    human_readable: bool
        If True, the state information is printed in a human interpretable form. By default, it is set to False.

    Returns
    -------
    Union[Tuple[List[TimeStep], List[int]], List[TimeStep]]
        The data generated from the interactions.
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
        if config.VERBOSE_LEVEL > 0:
            verba()
        state, action = mdp.random_step()
        if return_actions:
            actions.append(action)
        states.append(state)
        if state.last():
            if config.VERBOSE_LEVEL > 0:
                print("Last state:", state)
            state = mdp.reset()
            states.append(state)

    if return_actions:
        return states, actions
    return states


def prefixed_action_loop(
    mdp: Union["ContinuousMDP", "EpisodicMDP"],
    actions: Iterable[int],
    human_readable: bool = False,
) -> List[TimeStep]:
    """
    generates interaction with the MDP according to the actions given in input.

    Parameters
    ----------
    mdp: Union["ContinuousMDP", "EpisodicMDP"]
        The MDP instance.
    actions : Iterable[int]
        The actions to be selected.
    human_readable: bool
        If True, the state information is printed in a human interpretable form. By default, it is set to False.

    Returns
    -------
    Union[Tuple[List[TimeStep], List[int]], List[TimeStep]]
        The data generated from the interactions.
    """

    if human_readable:
        verba = lambda: print(mdp.get_grid_repr())
    else:
        verba = lambda: print("State:", state, "Action: ", action)

    states = []
    state = mdp.reset()
    states.append(state)
    for action in actions:
        if config.VERBOSE_LEVEL > 0:
            verba()
        state = mdp.step(action)
        states.append(state)
        if state.last():
            if config.VERBOSE_LEVEL > 0:
                print("Last state:", state)
            state = mdp.reset()
            states.append(state)
    return states
