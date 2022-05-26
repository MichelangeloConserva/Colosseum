from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum

# from functools import lru_cache
from multiprocessing import cpu_count
from typing import Any, Dict, Iterable, List, Tuple, Type, Union

import dm_env
import networkx as nx
import numpy as np
import yaml
from numpy.core._exceptions import _ArrayMemoryError
from pydtmc import MarkovChain
from scipy.stats import rv_continuous

from colosseum.agents.policy import ContinuousPolicy, EpisodicPolicy
from colosseum.dp.continuous import get_policy as continuous_policy
from colosseum.dp.continuous import get_random_policy as continuous_rp
from colosseum.dp.continuous import policy_evaluation as continuous_pe
from colosseum.dp.continuous import value_iteration as continuous_vi
from colosseum.dp.episodic import get_policy as episodic_policy
from colosseum.dp.episodic import get_random_policy as episodic_rp
from colosseum.dp.episodic import policy_evaluation as episodic_pe
from colosseum.dp.episodic import value_iteration as episodic_vi
from colosseum.experiments.hardness_reports import find_hardness_report_file
from colosseum.mdps import NodeType
from colosseum.mdps.markov_chain import get_markov_chain, transition_probabilities
from colosseum.mdps.mdp_classification import (
    MDPClass,
    get_continuous_MDP_class,
    get_episodic_MDP_class,
)
from colosseum.measures.diameter import (
    multi_thread_diameter_calculation,
    numba_continuous_diameter,
    single_thread_diameter_calculation,
)
from colosseum.measures.value_norm import (
    calculate_norm_average,
    calculate_norm_discounted,
)
from colosseum.utils.acme.specs import Array, DiscreteArray
from colosseum.utils.mdps import get_average_reward, get_average_rewards, mdp_string


class MDP(dm_env.Environment, ABC):
    """Base class for MDPs."""

    @staticmethod
    @abstractmethod
    def is_episodic() -> bool:
        """"""

    @staticmethod
    @abstractmethod
    def testing_parameters() -> Dict[str, Tuple]:
        return dict(
            seed=(99, 30),
            randomize_actions=(True,),
            lazy=(None, 0.1),
            random_action_p=(None, 0.1),
        )

    @staticmethod
    @abstractmethod
    def get_node_class() -> Type[Any]:
        """
        return the class of the nodes of the MDP.
        """

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """
        return the parameters of the MDP.
        """
        return dict(
            communication_type=self.communication_type.name,
            seed=self.seed,
            randomize_actions=self.randomize_actions,
            make_reward_stochastic=self.make_reward_stochastic,
            lazy=self.lazy,
            random_action_p=self.random_action_p,
        )

    @property
    @abstractmethod
    def num_actions(self) -> int:
        pass

    @property
    @abstractmethod
    def possible_starting_nodes(self) -> List[Any]:
        """
        returns a list containing all the possible starting node for this MDP instance.
        """

    # region Abstract private methods
    @abstractmethod
    def _calculate_next_nodes_prms(
        self, node: Any, action: int
    ) -> Tuple[Tuple[dict, float], ...]:
        """

        Parameters
        ----------
        node : NodeType
        action : int

        Returns
        -------
            the parameters of the possible next nodes reachable from the given node and action.
        """

    @abstractmethod
    def _calculate_reward_distribution(
        self, node: Any, action: IntEnum, next_node: Any
    ) -> rv_continuous:
        """

        Parameters
        ----------
        node : NodeType
        action : int
        next_node : NodeType

        Returns
        -------
        a continuous random distribution that is used to sample rewards for reaching next_node when selecting action in
        node.
        """

    @abstractmethod
    def _check_input_parameters(self):
        """
        Checks if the input parameters are valid.
        """
        assert self.random_action_p is None or (0 < self.random_action_p < 0.95)
        assert isinstance(self, EpisodicMDP) or isinstance(self, ContinuousMDP)

    @abstractmethod
    def _instantiate_starting_node_sampler(self) -> NextStateSampler:
        pass

    # endregion

    # region Abstract public methods
    @abstractmethod
    def calc_grid_repr(self, node: Any) -> np.array:
        """
        Produces a grid like representation of a given node of the MDP.
        """

    # endregion

    # region Magic methods
    def __str__(self):
        return mdp_string(type(self).__name__, self.summary)

    @abstractmethod
    def __init__(
        self,
        seed: int,
        randomize_actions: bool = True,
        lazy: float = None,
        random_action_p: float = None,
        r_max: float = 1.0,
        r_min: float = 0.0,
        force_single_thread=False,
        verbose=False,
        hardness_reports_folder="hardness_reports",
        **kwargs,
    ):
        """

        Parameters
        ----------
        seed : int
            the seed used for sampling rewards and next states.
        randomize_actions : bool, optional
            whether the effect of the actions changes for every node. It is particularly important to set this value to
             true when doing experiments to avoid immediately reaching highly rewarding states in some MDPs by just
             selecting the same action repeatedly. By default, it is set to true.
        lazy : float, optional
            the probability of an action not producing any effect on the MDP. By default, it is set to zero.
        random_action_p : float, optional
            the probability of selecting an action a random instead of the one specified by the agent. By default, it is
            set to zero.
        r_max : float, optional
            the maximum value of the reward. By default, it is set to one.
        r_min : float, optional
            the minimum value of the reward. By default, it is set to zero.
        force_single_thread : bool, optional
            checks whether the multiprocessing should be turned off.
        verbose: bool, optional
            checks whether to print verbose outputs.
        """
        self.last_starting_node = None
        self.cur_node = None
        self.last_edge = None
        self._are_transitions_deterministic = True
        self._are_rewards_deterministic = True
        self.necessary_reset = True
        self.h = 0
        self._action_mapping = dict()
        self.cached_rewards = dict()
        self._T = self._R = None
        self._gl = None
        self._ns = None
        self._rns = None
        self._summary = None
        self._gm = None
        self._hr = None
        self._ct = None
        self._moh = None
        self._sg = None
        self._sd = None
        self._d = None
        self._otp = None
        self._omc = None
        self._osd = None
        self._oars = None
        self._oar = None
        self._wars = None
        self._war = None
        self._rtp = None
        self._rmc = None
        self._rsd = None
        self._rars = None
        self._rar = None
        self._rn = None
        self._etmr = None
        self._cfetmr = None
        self._mv = None
        self._wp = None
        self._ov = dict()
        self._op = dict()
        self._rv = dict()
        self._rp = dict()
        self._wpsv = dict()
        self._opsv = dict()
        self._rpsv = dict()
        self._vnp = dict()
        self._rd = dict()
        self._eg = dict()

        self.random_action_p = None if random_action_p == 0.0 else random_action_p
        self.seed = seed
        self.lazy = lazy
        self._rng = np.random.RandomState(seed)
        self._fast_rng = random.Random(seed)
        self.randomize_actions = randomize_actions
        self.r_min = r_min
        self.r_max = r_max
        self.force_single_thread = force_single_thread or cpu_count() < 4
        self.verbose = verbose
        self.hardness_reports_folder = hardness_reports_folder

        self.node_class = self.get_node_class()
        self._check_input_parameters()
        self.starting_node_sampler = self._instantiate_starting_node_sampler()

        self.G = nx.DiGraph()

        self._instantiate_mdp()

        self._node_to_index = dict()
        self._index_to_node = dict()
        for i, node in enumerate(self.G.nodes):
            # Store index for fast retrieving
            self._node_to_index[node] = i
            self._index_to_node[i] = node

    # endregion

    # region Properties
    @property
    def worst_policy(self) -> Union["ContinuousPolicy", "EpisodicPolicy"]:
        if self._wp is None:
            T, R = self.transition_matrix_and_rewards
            if self.is_episodic():
                self._wp = episodic_policy(
                    episodic_vi(self.H, T, -R)[0], self._fast_rng
                )
            else:
                self._wp = continuous_policy(continuous_vi(T, -R)[0], self._fast_rng)
        return self._wp

    @property
    def worst_value(self) -> np.ndarray:
        if self._mv is None:
            T, R = self.transition_matrix_and_rewards
            if self.is_episodic():
                _, self._mv = episodic_pe(self.H, T, R, self.worst_policy.pi_matrix)
            else:
                _, self._mv = continuous_pe(T, R, self.worst_policy.pi_matrix)
        return self._mv

    @property
    def _graph_layout(self) -> Dict[NodeType, Tuple[float, float]]:
        return nx.nx_agraph.graphviz_layout(self.G)

    @property
    def graph_layout(self) -> Dict[NodeType, Tuple[float, float]]:
        """
        returns and caches the graph layout for the graph underling the MDP.
        """
        if self._gl is None:
            self._gl = self._graph_layout
        return self._gl

    @property
    def are_transitions_deterministic(self) -> bool:
        return self._are_transitions_deterministic

    @property
    def are_rewards_deterministic(self) -> bool:
        return self._are_rewards_deterministic

    @property
    def num_states(self) -> int:
        if self._ns is None:
            self._ns = len(self.G.nodes)
        return self._ns

    @property
    def starting_nodes(self) -> List[NodeType]:
        return self.starting_node_sampler.next_states

    @property
    def transition_matrix_and_rewards(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._T is None:
            self._T = np.zeros(
                (self.num_states, self.action_spec().num_values, self.num_states),
                dtype=np.float32,
            )
            self._R = np.zeros(
                (self.num_states, self.action_spec().num_values), dtype=np.float32
            )
            for i, node in enumerate(self.G.nodes()):
                for action, td in self.get_info_class(
                    node
                ).transition_distributions.items():
                    r = 0
                    for next_state, prob in zip(td.next_states, td.probs):
                        self._T[i, action, self.node_to_index(next_state)] += prob
                        r += (
                            prob
                            * self.get_reward_distribution(
                                node, action, next_state
                            ).mean()
                        )
                    self._R[i, action] = r
                assert np.isclose(self._T[i, action].sum(), 1)
            assert np.isnan(self._R).sum() == 0
        return self._T, self._R

    def action_mapping(self, node) -> Tuple:
        """
        creates a randomized action mapping between node and all the available actions.

        Parameters
        ----------
        node : NodeType

        Returns
        -------
        a tuple representing a mapping from actions to randomized actions.
        """
        if node not in self._action_mapping:
            self._action_mapping[node] = (
                self._rng.rand(self.num_actions).argsort().tolist()
                if self.randomize_actions
                else list(range(self.num_actions))
            )
        return self._action_mapping[node]

    @property
    def recurrent_nodes_set(self) -> List[Any]:
        """
        calculates the set of nodes which are recurrent for weakly communicating MDPs. For all other types of MDPs, it
        returns the entire state set.

        Returns
        -------
        a list containing the nodes in the recurrent class.
        """
        if self._rns is None:
            if self.communication_type == MDPClass.WEAKLY_COMMUNICATING:
                c = nx.condensation(self.G)
                leaf_nodes = [x for x in c.nodes() if c.out_degree(x) == 0]
                assert len(leaf_nodes) == 1
                self._rns = c.nodes(data="members")[leaf_nodes[0]]
            elif self.communication_type == MDPClass.NON_WEAKLY_COMMUNICATING:
                raise ValueError(
                    "A non weakly communicating class does not have a recurrent nodes set"
                )
            else:
                self._rns = self.G.nodes
        return self._rns

    @property
    def summary(self) -> Dict[str, Dict[str, Any]]:
        """
        produces a summary of the MDP instance.
        """
        from colosseum.utils.logging import clean_for_storing

        if self._summary is None:
            self._summary = {
                "Parameters": clean_for_storing(self.parameters),
                "Measure of hardness": clean_for_storing(self.measures_of_hardness),
                "Graph metrics": clean_for_storing(self.graph_metrics),
            }
        return self._summary

    @property
    def graph_metrics(self) -> Dict[str, Union[int, float]]:
        """
        returns a dictionary with metric name as a key and metric value as item. For now, it only returns number of
        nodes and number of edges but it can be expanded using the networkx functions.
        """
        if self._gm is None:
            G = self.episodic_graph() if self.is_episodic() else self.G
            d = dict()
            d["# nodes"] = len(G.nodes)
            d["# edges"] = len(G.edges)
            self._gm = d
        return self._gm

    @property
    def hardness_report(self) -> Union[Dict, None]:
        if self._hr is None:
            report_file = find_hardness_report_file(self, self.hardness_reports_folder)
            if report_file:
                with open(report_file, "r") as f:
                    report = yaml.load(f, yaml.Loader)
                self._hr = report
            else:
                self._hr = False
        if self._hr:
            return self._hr
        return None

    @property
    def communication_type(self: Union["EpisodicMDP", "ContinuousMDP"]) -> MDPClass:
        """
        calculates anc caches the communication class of the MDP instance.
        """
        if self._ct is None:
            if self.is_episodic():
                T, _ = self.episodic_transition_matrix_and_rewards
                self._ct = get_episodic_MDP_class(
                    T, self.episodic_graph(True), self.verbose
                )
            T, _ = self.transition_matrix_and_rewards
            self._ct = get_continuous_MDP_class(T)
        return self._ct

    # region Measures of hardness
    @property
    def measures_of_hardness(self) -> Dict[str, float]:
        """
        calculates and cached the measure of hardness for the MDP instance.
        """
        if self._moh is None:
            self._moh = dict(
                diameter=self.diameter,
                suboptimal_gaps=self.suboptimal_gaps,
                value_norm=self.get_value_norm_policy(True),
            )
        return self._moh

    @property
    def suboptimal_gaps(self: Union["EpisodicMDP", "ContinuousMDP"]) -> float:
        """
        calculates and caches the average of the reciprocal of the sub-optimality gaps.
        """
        if self._sg is None:
            if self.hardness_report:
                self._sg = self.hardness_report["MDP measure of hardness"][
                    "suboptimal_gaps"
                ]
            else:
                Q, V = self.optimal_value()
                gaps = V[..., None] - Q

                if self.is_episodic():
                    gaps = np.vstack([gaps[h, s] for h, s in self.reachable_nodes])
                gaps = gaps[~np.isclose(gaps, 0, 1e-3, 1e-3)] + 0.1
                self._sg = (1 / gaps).sum()
        return self._sg

    @property
    def starting_distribution(self) -> np.ndarray:
        if self._sd is None:
            self._sd = np.array(
                [self.starting_node_sampler.prob(n) for n in self.G.nodes]
            )
        return self._sd

    @property
    def diameter(self: Union["EpisodicMDP", "ContinuousMDP"]) -> float:
        """
        calculates and caches the diameter of the MDP.
        """
        if self._d is None:

            if self.hardness_report:
                self._d = self.hardness_report["MDP measure of hardness"]["diameter"]
            else:
                # TODO: this is slower in some cases due to the non parallelization.
                # if self.are_transitions_deterministic:
                #     return deterministic_diameter(
                #         self.episodic_graph(True) if self.is_episodic() else self.G,
                #         self.verbose,
                #     )

                T, _ = self.transition_matrix_and_rewards
                if self.is_episodic():
                    T, _ = self.episodic_transition_matrix_and_rewards

                    if self.force_single_thread:
                        self._d = single_thread_diameter_calculation(
                            T, verbose=self.verbose
                        )
                    else:
                        self._d = multi_thread_diameter_calculation(
                            T, verbose=self.verbose
                        )
                elif self.communication_type == MDPClass.NON_WEAKLY_COMMUNICATING:
                    self._d = np.inf
                else:
                    if self.communication_type == MDPClass.WEAKLY_COMMUNICATING:
                        # The diameter is calculated only on the recurrent class
                        node_indices = list(
                            map(self.node_to_index, self.recurrent_nodes_set)
                        )
                        T = T[
                            np.ix_(
                                node_indices, [True] * self.num_actions, node_indices
                            )
                        ]
                    self._d = numba_continuous_diameter(
                        T,
                        self.force_single_thread or self.num_states < 50,
                        verbose=self.verbose,
                    )
        return self._d

    def get_value_norm_policy(
        self: Union["EpisodicMDP", "ContinuousMDP"],
        is_discounted: bool,
        policy: ContinuousPolicy = None,
    ) -> float:
        """
        Calculates and caches the environmental value norm.

        Parameters
        ----------
        is_discounted : bool
            check whether to calculate the value norm in the discounted or undiscounted setting.
        policy : ContinuousPolicy, optional
            the policy for which the norm is calculated. By default, it is set to the optimal policy.
        """
        if not (is_discounted, policy) in self._vnp:
            if self.hardness_report:
                self._vnp[is_discounted, policy] = self.hardness_report[
                    "MDP measure of hardness"
                ]["value_norm"]
            elif self.are_transitions_deterministic and self.are_rewards_deterministic:
                self._vnp[is_discounted, policy] = 0.0
            else:
                if self.is_episodic():
                    T, R = self.continuous_form_episodic_transition_matrix_and_rewards
                else:
                    T, R = self.transition_matrix_and_rewards

                if policy is None:
                    if is_discounted:
                        if self.is_episodic():
                            _, V = continuous_vi(T, R)
                        else:
                            _, V = self.optimal_value()
                        self._vnp[is_discounted, policy] = calculate_norm_discounted(
                            T, V
                        )
                    else:
                        self._vnp[is_discounted, policy] = calculate_norm_average(
                            T,
                            self.optimal_transition_probs(),
                            self.optimal_average_rewards(),
                            verbose=self.verbose,
                        )
                else:
                    if is_discounted:
                        V = continuous_pe(
                            T, R, policy.pi_matrix, gamma=0.99, epsilon=1e-7
                        )
                        self._vnp[is_discounted, policy] = calculate_norm_discounted(
                            T, V
                        )
                    else:
                        tps = transition_probabilities(T, policy)

                        ars = get_average_rewards(R, policy)
                        self._vnp[is_discounted, policy] = calculate_norm_average(
                            T, tps, ars, self.verbose
                        )
        return self._vnp[is_discounted, policy]

    # endregion

    # region Private methods
    def _next_seed(self) -> int:
        """
        returns a new fast random seed.
        """
        return self._fast_rng.randint(0, 10_000)

    def _instantiate_mdp(self):
        """
        recursively instantiate the MDP.
        """
        for sn in self.starting_nodes:
            self._instantiate_transitions(sn)

    def _transition(self, next_states, probs, node, action, next_node, p):
        next_states.append(next_node)
        probs.append(p)
        if (
            self._are_rewards_deterministic
            and self.get_reward_distribution(node, action, next_node).dist.name
            != "deterministic"
        ):
            self._are_rewards_deterministic = False
        self.G.add_edge(node, next_node)

    def _instantiate_transition(self, node: NodeType, action: int) -> NextStateSampler:
        next_states = []
        probs = []
        for next_node, p in self._get_next_node(node, action):
            p1_lazy = 1.0 if self.lazy is None else (1 - self.lazy)
            p = p1_lazy * p
            p = (
                p
                if self.random_action_p is None
                else (
                    (1 - self.random_action_p) * p
                    + p * self.random_action_p / self.num_actions
                )
            )
            self._transition(next_states, probs, node, action, next_node, p)
        if self.lazy is not None:
            next_node = self._get_lazy_node(node)
            self._transition(next_states, probs, node, action, next_node, self.lazy)
        if self.random_action_p is not None:
            for a in range(self.num_actions):
                if a == action:
                    continue
                for next_node, p in self._get_next_node(node, a):
                    p = p1_lazy * self.random_action_p * p / self.num_actions
                    # p = p1_lazy * p
                    self._transition(next_states, probs, node, action, next_node, p)

        assert np.isclose(sum(probs), 1.0)

        return NextStateSampler(
            next_states=next_states,
            probs=probs,
            seed=self._next_seed(),
        )

    def _instantiate_transitions(self, node):
        if not self.G.has_node(node) or len(list(self.G.successors(node))) == 0:
            transition_distributions = dict()
            for a in range(self.num_actions):
                td = self._instantiate_transition(node, a)

                if not td.is_deterministic:
                    self._are_transitions_deterministic = False

                for ns in td.next_states:
                    self._instantiate_transitions(ns)
                transition_distributions[self.real_action_to_randomized(node, a)] = td

            assert all(
                action in transition_distributions.keys()
                for action in range(self.num_actions)
            )
            self.add_node_info_class(node, transition_distributions)

    # endregion

    # region Public methods
    def get_grid_repr(self) -> np.array:
        return self.calc_grid_repr(self.cur_node)

    def action_spec(self) -> DiscreteArray:
        """
        returns the action spec of the environment.
        """
        return DiscreteArray(self.num_actions, name="action")

    def real_action_to_randomized(self, node, action):
        return self.action_mapping(node)[action]

    def observation_spec(self) -> Array:
        """
        returns the observation spec of the environment.
        """
        return DiscreteArray(num_values=self.num_states, name="observation")

    def get_transition_distributions(
        self, node: NodeType
    ) -> Dict[int, Union[rv_continuous, NextStateSampler]]:
        return self.get_info_class(node).transition_distributions

    def get_info_class(self, n: NodeType) -> NodeInfoClass:
        """
        returns the container class (NodeInfoClass) associated with node n.
        """
        return self.G.nodes[n]["info_class"]

    def add_node_info_class(
        self, n: NodeType, transition_distributions: Dict[int, NextStateSampler]
    ):
        """
        add a container class (NodeInfoClass) in the node n containing the transition distributions.

        Parameters
        ----------
        n : NodeType
            the node to which add the NodeInfoClass
        transition_distributions : Dict[int, NextStateSampler]
            the dictionary containing the transition distributions.
        """
        self.G.nodes[n]["info_class"] = NodeInfoClass(
            transition_distributions=transition_distributions,
            actions_visitation_count=dict.fromkeys(range(self.num_actions), 0),
        )

    def node_to_index(self, node: NodeType) -> int:
        return self._node_to_index[node]

    def index_to_node(self, index: int) -> NodeType:
        return self._index_to_node[index]

    def get_reward_distribution(
        self, node: NodeType, action: int, next_node: NodeType
    ) -> rv_continuous:
        """
        returns and caches the reward distribution.
        """
        if (node, action, next_node) not in self._rd:
            self._rd[node, action, next_node] = self._calculate_reward_distribution(
                node, self.real_action_to_randomized(node, action), next_node
            )
        return self._rd[node, action, next_node]

    def sample_reward(
        self, node: NodeType, action: Union[int, IntEnum], next_node: NodeType
    ) -> float:
        """
        returns a sample from the reward distribution.
        """
        if (node, action, next_node) not in self.cached_rewards or len(
            self.cached_rewards[node, action, next_node]
        ) == 0:
            self.cached_rewards[node, action, next_node] = (
                self.get_reward_distribution(node, action, next_node).rvs(5000).tolist()
            )
        r = self.cached_rewards[node, action, next_node].pop(0)
        # r = self.get_reward_distribution(node, action, next_node).rvs()
        return r * (self.r_max - self.r_min) - self.r_min

    def reset(self) -> dm_env.TimeStep:
        """
        resets the environment to a starting node.
        """
        self.necessary_reset = False
        self.h = 0
        self.cur_node = self.last_starting_node = self.starting_node_sampler.sample()
        node_info_class = self.get_info_class(self.cur_node)
        node_info_class.update_visitation_counts()
        return dm_env.restart(self.node_to_index(self.cur_node))

    def step(self, action: int) -> dm_env.TimeStep:
        """
        Takes a step in the MDP given an action.
        """

        self.h += 1
        assert not self.necessary_reset

        # In case the action is a numpy array
        action = int(action)

        # Moving the current node according to the action played
        old_node = self.cur_node
        self.cur_node = self.get_info_class(old_node).sample_next_state(action)
        self.last_edge = old_node, self.cur_node
        node_info_class = self.get_info_class(self.cur_node)
        node_info_class.update_visitation_counts(action)

        # Calculate reward and observation
        reward = self.sample_reward(old_node, action, self.cur_node)
        observation = self.node_to_index(self.cur_node)

        # Wrapping the time step in a dm_env.TimeStep
        if isinstance(self, EpisodicMDP) and self.h >= self.H:
            self.necessary_reset = True
            return dm_env.termination(
                reward=reward, observation=self.observation_spec().generate_value() - 1
            )
        return dm_env.transition(reward=reward, observation=observation)

    def random_step(self, auto_reset=False) -> Tuple[dm_env.TimeStep, int]:
        action = int(self._rng.randint(self.action_spec().num_values))
        ts = self.step(action)
        if auto_reset and ts.last():
            self.reset()
        return ts, action

    # endregion

    def _get_next_node(self, node, action) -> List[Any, float]:
        return [
            (self.node_class(**node_prms), prob)
            for node_prms, prob in self._calculate_next_nodes_prms(node, action)
        ]

    def _get_lazy_node(self, node: NodeType) -> NodeType:
        return node

    def get_visitation_counts(self, state_only=True) -> Dict[NodeType, int]:
        if state_only:
            return {
                node: self.get_info_class(node).state_visitation_count
                for node in self.G.nodes
            }
        return {
            (node, a): self.get_info_class(node).actions_visitation_count[a]
            for node in self.G.nodes
            for a in range(self.num_actions)
        }

    def get_value_node_labels(
        self: Union["ContinuousMDP", "EpisodicMDP"], V: np.ndarray = None
    ) -> Dict[NodeType, float]:
        if V is None:
            _, V = self.optimal_value()
        else:
            if isinstance(self, EpisodicMDP):
                h, d = V.shape
                assert h == self.H and d == self.num_states
            else:
                assert len(V) == self.num_states
        return {
            node: np.round(
                V[0, self.node_to_index(node)]
                if self.is_episodic()
                else V[self.node_to_index(node)],
                2,
            )
            for node in self.G.nodes
        }


class ContinuousMDP(MDP, ABC):
    """Base class for continuous MDPs."""

    @staticmethod
    def is_episodic() -> bool:
        return False

    def optimal_value(self) -> Tuple[np.ndarray, np.ndarray]:
        if 0 not in self._ov:
            self._ov[0] = continuous_vi(*self.transition_matrix_and_rewards)
        return self._ov[0]

    def optimal_policy(self) -> ContinuousPolicy:
        if 0 not in self._op:
            self._op[0] = continuous_policy(self.optimal_value()[0], self._fast_rng)
        return self._op[0]

    def optimal_transition_probs(self) -> np.ndarray:
        if self._otp is None:
            T, _ = self.transition_matrix_and_rewards
            self._otp = transition_probabilities(T, self.optimal_policy())
        return self._otp

    def optimal_markov_chain(self) -> MarkovChain:
        if self._omc is None:
            self._omc = get_markov_chain(self.optimal_transition_probs())
        return self._omc

    def optimal_stationary_distribution(self) -> np.array:
        if self._osd is None:
            self._osd = self.optimal_markov_chain().pi[0]
        return self._osd

    def optimal_average_rewards(self) -> np.array:
        if self._oars is None:
            _, R = self.transition_matrix_and_rewards
            self._oars = get_average_rewards(R, self.optimal_policy())
        return self._oars

    def optimal_average_reward(self) -> float:
        if self._oar is None:
            T, _ = self.transition_matrix_and_rewards

            self._oar = get_average_reward(
                self.optimal_average_rewards(), T, self.optimal_policy()
            )
        return self._oar

    def worst_average_rewards(self) -> np.array:
        if self._wars is None:
            _, R = self.transition_matrix_and_rewards
            self._wars = get_average_rewards(R, self.worst_policy)
        return self._wars

    def worst_average_reward(self) -> float:
        if self._war is None:
            T, _ = self.transition_matrix_and_rewards
            self._war = get_average_reward(
                self.worst_average_rewards(), T, self.worst_policy
            )
        return self._war

    def random_policy(self) -> ContinuousPolicy:
        if 0 not in self._rp:
            self._rp[0] = continuous_rp(self.num_states, self.num_actions)
        return self._rp[0]

    def random_value(self) -> Tuple[np.ndarray, np.ndarray]:
        if 0 not in self._rv:
            self._rv[0] = continuous_pe(
                *self.transition_matrix_and_rewards, self.random_policy().pi_matrix
            )
        return self._rv[0]

    def random_transition_probs(self) -> np.ndarray:
        if self._rtp is None:
            T, _ = self.transition_matrix_and_rewards
            self._rtp = transition_probabilities(T, self.random_policy())
        return self._rtp

    def random_markov_chain(self) -> MarkovChain:
        if self._rmc is None:
            self._rmc = get_markov_chain(self.random_transition_probs())
        return self._rmc

    def random_stationary_distribution(self) -> np.array:
        if self._rsd is None:
            self._rsd = self.random_markov_chain().pi[0]
        return self._rsd

    def random_average_rewards(self) -> np.array:
        if self._rars is None:
            _, R = self.transition_matrix_and_rewards
            self._rars = get_average_rewards(R, self.random_policy())
        return self._rars

    def random_average_reward(self) -> float:
        if self._rar is None:
            T, _ = self.transition_matrix_and_rewards
            self._rar = get_average_reward(
                self.random_average_rewards(), T, self.random_policy()
            )
        return self._rar

    def __init__(self, *args, **kwargs):
        super(ContinuousMDP, self).__init__(*args, **kwargs)

        T, _ = self.transition_matrix_and_rewards
        for i in range(self.num_states):
            assert (
                T[..., i].max() > 0.0
            ), "Some states in the MDP are not reachable by any other state."


class EpisodicMDP(MDP, ABC):
    """Base class for episodic MDPs."""

    @property
    def reachable_nodes(self) -> List[NodeType]:
        if self._rn is None:
            self._rn = [
                (h, self.node_to_index(n)) for h, n in self.episodic_graph().nodes
            ]
        return self._rn

    def optimal_value(self, continuous_form=False) -> Tuple[np.ndarray, np.ndarray]:
        if continuous_form not in self._ov:
            if continuous_form:
                self._ov[continuous_form] = continuous_vi(
                    *self.continuous_form_episodic_transition_matrix_and_rewards
                )
            else:
                self._ov[continuous_form] = episodic_vi(
                    self.H, *self.transition_matrix_and_rewards
                )
        return self._ov[continuous_form]

    def optimal_policy(
        self, continuous_form=False
    ) -> Union[EpisodicPolicy, ContinuousPolicy]:
        if continuous_form not in self._op:
            if continuous_form:
                self._op[continuous_form] = continuous_policy(
                    self.optimal_value(True)[0], self._fast_rng
                )
            else:
                self._op[continuous_form] = episodic_policy(
                    self.optimal_value()[0], self._fast_rng
                )
        return self._op[continuous_form]

    def optimal_transition_probs(self) -> np.ndarray:
        if self._otp is None:
            T, _ = self.continuous_form_episodic_transition_matrix_and_rewards
            self._otp = transition_probabilities(T, self.optimal_policy(True))
        return self._otp

    def optimal_markov_chain(self) -> MarkovChain:
        if self._omc is None:
            self._omc = get_markov_chain(self.optimal_transition_probs())
        return self._omc

    def optimal_stationary_distribution(self) -> np.array:
        if self._osd is None:
            self._osd = self.optimal_markov_chain().pi[0]
        return self._osd

    def optimal_average_rewards(self) -> np.array:
        if self._oars is None:
            _, R = self.continuous_form_episodic_transition_matrix_and_rewards
            self._oars = get_average_rewards(R, self.optimal_policy(True))
        return self._oars

    def optimal_average_reward(self) -> float:
        if self._oar is None:
            oar = 0.0
            for sn, p in self.starting_node_sampler.next_states_and_probs:
                oar += p * self.optimal_policy_starting_value(sn)
            self._oar = oar / self.H
        return self._oar

    def random_policy(
        self, continuous_form=False
    ) -> Union[EpisodicPolicy, ContinuousPolicy]:
        if continuous_form not in self._rp:
            if continuous_form:
                T, _ = self.continuous_form_episodic_transition_matrix_and_rewards
                self._rp[continuous_form] = continuous_rp(len(T), self.num_actions)
            else:
                self._rp[continuous_form] = episodic_rp(
                    self.H, self.num_states, self.num_actions
                )
        return self._rp[continuous_form]

    def random_value(self, continuous_form=False) -> Tuple[np.ndarray, np.ndarray]:
        if continuous_form not in self._rv:
            rp = self.random_policy(continuous_form)
            if continuous_form:
                self._rv[continuous_form] = continuous_pe(
                    *self.continuous_form_episodic_transition_matrix_and_rewards,
                    rp.pi_matrix,
                )
            else:
                self._rv[continuous_form] = episodic_pe(
                    self.H,
                    *self.transition_matrix_and_rewards,
                    self.random_policy().pi_matrix,
                )
        return self._rv[continuous_form]

    def random_transition_probs(self) -> np.ndarray:
        if self._rtp is None:
            T, _ = self.continuous_form_episodic_transition_matrix_and_rewards
            self._rtp = transition_probabilities(T, self.random_policy())
        return self._rtp

    def random_markov_chain(self) -> MarkovChain:
        if self._rmc is None:
            self._rmc = get_markov_chain(self.random_transition_probs())
        return self._rmc

    def random_stationary_distribution(self) -> np.array:
        if self._rsd is None:
            self._rsd = self.random_markov_chain().pi[0]
        return self._rsd

    def random_average_rewards(self) -> np.array:
        if self._rars is None:
            _, R = self.continuous_form_episodic_transition_matrix_and_rewards
            self._rars = get_average_rewards(R, self.random_policy(True))
        return self._rars

    def random_average_reward(self) -> float:
        if self._rar is None:
            oar = 0.0
            for sn, p in self.starting_node_sampler.next_states_and_probs:
                oar += p * self.random_policy_starting_value(sn)
            self._rar = oar / self.H
        return self._rar

    def optimal_policy_starting_value(self, sn: NodeType) -> float:
        if sn not in self._opsv:
            _, V = self.optimal_value()
            self._opsv[sn] = V[0, self.node_to_index(sn)]
        return self._opsv[sn]

    def random_policy_starting_value(self, sn) -> float:
        if sn not in self._rpsv:
            _, V = self.random_value()
            self._rpsv[sn] = V[0, self.node_to_index(sn)]
        return self._rpsv[sn]

    def worst_policy_starting_value(self, sn) -> float:
        if sn not in self._wpsv:
            self._wpsv[sn] = self.worst_value[0, self.node_to_index(sn)]
        return self._wpsv[sn]

    @property
    def episodic_transition_matrix_and_rewards(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        returns the |H| x |S| x |A| x |S| episodic transition matrix and the |H| x |S| x |A| matrix of expected rewards.
        """
        if self._etmr is None:
            T, R = self.transition_matrix_and_rewards
            T_epi = np.zeros(
                (self.H, self.num_states, self.num_actions, self.num_states),
                dtype=np.float32,
            )
            for sn, p in self.starting_node_sampler.next_states_and_probs:
                sn = self.node_to_index(sn)
                T_epi[0, sn] = T[sn]
                T_epi[self.H - 1, :, :, sn] = p
            for h in range(1, self.H - 1):
                for s in range(len(T)):
                    if T_epi[h - 1, :, :, s].sum() > 0:
                        T_epi[h, s] = T[s]
            R = np.tile(R, (self.H, 1, 1))
            R[-1] = 0.0
            self._etmr = T_epi, R
        return self._etmr

    def episodic_graph(self, remove_label=False) -> nx.DiGraph:
        """
        returns the graph of the MDP augmented with the time step in the state space.
        """
        if remove_label not in self._eg:
            G = nx.DiGraph()

            def add_successors(n, h):
                n_ = self.node_to_index(n) if remove_label else n
                if h < self.H - 1:
                    successors = self.G.successors(n)
                else:
                    successors = self.starting_node_sampler.next_states
                for succ in successors:
                    succ_ = self.node_to_index(succ) if remove_label else succ
                    next_h = (h + 1) if h + 1 != self.H else 0
                    G.add_edge((h, n_), (next_h, succ_))
                    if h < self.H - 1 and len(list(G.successors((next_h, succ_)))) == 0:
                        add_successors(succ, next_h)

            for sn in self.starting_node_sampler.next_states:
                add_successors(sn, 0)

            self._eg[remove_label] = G
        return self._eg[remove_label]

    @property
    def continuous_form_episodic_transition_matrix_and_rewards(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        return the transition matrix and the expected rewards matrix for the infinite horizon MDP obtained by augmenting
        the state space with the time step.
        """
        if self._cfetmr is None:

            G = self.episodic_graph(True)
            T, R = self.transition_matrix_and_rewards

            try:
                T_epi = np.zeros(
                    (len(G.nodes), self.num_actions, len(G.nodes)), np.float32
                )
                R_epi = np.zeros((len(G.nodes), self.num_actions), np.float32)
            except _ArrayMemoryError:
                raise ValueError(
                    "It is not possible calculate the value for this MDP. Its continuous form is too large."
                )

            nodes = list(G.nodes)
            for h, n in G.nodes:
                if h == self.H - 1:
                    for sn, p in self.starting_node_sampler.next_states_and_probs:
                        T_epi[nodes.index((h, n)), :, self.node_to_index(sn)] = p
                        R_epi[nodes.index((h, n))] = R[n]
                else:
                    for hp1, nn in G.successors((h, n)):
                        T_epi[nodes.index((h, n)), :, nodes.index((hp1, nn))] = T[
                            n, :, nn
                        ]
                        R_epi[nodes.index((h, n))] = R[n]

            assert np.isclose(T_epi.sum(-1), 1.0).all()
            self._cfetmr = T_epi, R_epi
        return self._cfetmr

    @staticmethod
    def is_episodic() -> bool:
        return True

    def calculate_H(self, H: int):
        """
        calculates a meaningful minimal horizon for the MDP.
        """

        if "Taxi" in str(type(self)):
            # it is complicated to give the same horizon to different seed of the same MDP instance
            # for the Taxi MDP
            minimal_H = int(1.5 * self.size ** 2)
        else:
            minimal_H = (
                max(
                    max(nx.shortest_path_length(self.G, sn).values())
                    for sn in self.possible_starting_nodes
                )
                + 1
            )
        if H is None:
            self.H = minimal_H
        else:
            self.H = max(minimal_H, H)

    def __init__(self, H: int = None, **kwargs):
        super(EpisodicMDP, self).__init__(**kwargs)

        self.calculate_H(H)

        T, _ = self.episodic_transition_matrix_and_rewards
        for i in range(self.num_states):
            assert (
                T[..., i].max() > 0.0
            ), "Some states in the MDP are not reachable by any other state."

    @property
    def parameters(self) -> Dict[str, Any]:
        parameters = super(EpisodicMDP, self).parameters
        parameters["H"] = self.H
        return parameters

    def get_grid_repr(self) -> np.array:
        grid = self.calc_grid_repr(self.cur_node)
        while grid.shape[1] < 2 + len(str(self.h)):
            adder = np.zeros((grid.shape[1], 1), dtype=str)
            adder[:] = "X"
            grid = np.hstack((grid, adder))
        title = np.array(
            [" "] * grid.shape[1] + ["_"] * grid.shape[1], dtype=str
        ).reshape(2, -1)
        title[0, 0] = "H"
        title[0, 1] = "="
        for i, l in enumerate(str(self.h)):
            title[0, 2 + i] = l
        return np.vstack((title, grid))


@dataclass()
class NodeInfoClass:
    transition_distributions: Dict[int, Union[rv_continuous, NextStateSampler]]
    actions_visitation_count: Dict[int, int]
    state_visitation_count: int = 0

    def update_visitation_counts(self, action: int = None):
        self.state_visitation_count += 1
        if action is not None:
            self.actions_visitation_count[action] += 1

    def sample_next_state(self, action: int):
        return self.transition_distributions[action].sample()


class NextStateSampler:
    @property
    def next_states_and_probs(self) -> Iterable[Tuple[NodeType, float]]:
        return zip(self.next_states, self.probs)

    def __init__(self, next_states: list, seed: int = None, probs: List[float] = None):
        assert len(next_states) > 0
        self.next_states = next_states
        self._probs = dict()

        # Deterministic sampler
        if len(next_states) == 1:
            assert probs is None or len(probs) == 1
            self.next_state = next_states[0]
            self.probs = [1.0]
            self.is_deterministic = True
        # Stochastic sampler
        else:
            assert seed is not None
            self.probs = probs
            self._rng = random.Random(seed)
            self.n = len(next_states)
            self.is_deterministic = False
            self.cached_states = self._rng.choices(
                self.next_states, weights=self.probs, k=5000
            )

    def sample(self) -> NodeType:
        """:returns a sample of the next state distribution."""
        if self.is_deterministic:
            return self.next_state
        if len(self.cached_states) == 0:
            self.cached_states = self._rng.choices(
                self.next_states, weights=self.probs, k=5000
            )
        return self.cached_states.pop(0)

    def mode(self) -> NodeType:
        """:returns the most probable next state."""
        if self.is_deterministic:
            return self.next_state
        return self.next_states[np.argmax(self.probs)]

    def prob(self, n: NodeType) -> float:
        if n not in self._probs:
            if n not in self.next_states:
                self._probs[n] = 0.0
            else:
                self._probs[n] = self.probs[self.next_states.index(n)]
        return self._probs[n]
