import abc
import os
import random
import sys
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Tuple, Type, Union

import dm_env
import networkx as nx
import numpy as np
import yaml
from dm_env.specs import BoundedArray, Array
from pydtmc import MarkovChain
from scipy.stats import rv_continuous

from colosseum.dynamic_programming import (
    discounted_policy_iteration,
    discounted_value_iteration,
)
from colosseum.dynamic_programming.infinite_horizon import discounted_policy_evaluation
from colosseum.dynamic_programming.utils import get_policy_from_q_values
from colosseum.hardness.measures import (
    calculate_norm_average,
    calculate_norm_discounted,
    find_hardness_report_file,
    get_diameter,
    get_sum_reciprocals_suboptimality_gaps,
)
from colosseum.mdp.utils.communication_class import (
    MDPCommunicationClass,
    get_communication_class,
    get_recurrent_nodes_set,
)
from colosseum.mdp.utils.custom_samplers import NextStateSampler
from colosseum.mdp.utils.markov_chain import (
    get_average_rewards,
    get_markov_chain,
    get_stationary_distribution,
    get_transition_probabilities,
)
from colosseum.mdp.utils.mdp_creation import (
    NodeInfoClass,
    get_transition_matrix_and_rewards,
    instantiate_transitions,
)
from colosseum.mdp.utils.state_representation import (
    StateRepresentationType,
    RepresentationMapping,
)
from colosseum.utils import clean_for_storing
from colosseum.utils.acme.specs import DiscreteArray
from colosseum.utils.formatter import clean_for_file_path

if TYPE_CHECKING:
    from colosseum.mdp import ACTION_TYPE, NODE_TYPE, ContinuousMDP, EpisodicMDP

sys.setrecursionlimit(5000)


class BaseMDP(dm_env.Environment, abc.ABC):
    @staticmethod
    def get_available_hardness_measures() -> List[str]:
        return ["diameter", "value_norm", "suboptimal_gaps"]

    @staticmethod
    @abc.abstractmethod
    def does_seed_change_MDP_structure() -> bool:
        """
        returns True if when changing the seed the transition matrix and/or rewards matrix change. This for example may
        happen when there are fewer starting states that possible one and the effective starting states are picked
        randomly based on the seed.
        """

    @staticmethod
    @abc.abstractmethod
    def is_episodic() -> bool:
        """
        returns whether the MDP is episodic.
        """

    @staticmethod
    @abc.abstractmethod
    def sample_parameters(n: int, seed: int = None) -> List[Dict[str, Any]]:
        """
        returns n sampled parameters that can be used to construct an MDP in a reasonable amount of time.
        """

    @staticmethod
    @abc.abstractmethod
    def _sample_parameters(
        n: int, is_episodic: bool, seed: int = None
    ) -> List[Dict[str, Any]]:
        """
        returns n sampled parameters that can be used to construct an MDP in a reasonable amount of time.
        """

    @staticmethod
    @abc.abstractmethod
    def get_node_class() -> Type["NODE_TYPE"]:
        """
        returns the class of the nodes of the MDP.
        """

    @property
    @abc.abstractmethod
    def n_actions(self) -> int:
        """
        returns the number of available actions.
        """

    @abc.abstractmethod
    def _get_next_nodes_parameters(
        self, node: "NODE_TYPE", action: "ACTION_TYPE"
    ) -> Tuple[Tuple[dict, float], ...]:
        """
        returns the parameters of the possible next nodes reachable from the given node and action.
        """

    @abc.abstractmethod
    def _get_reward_distribution(
        self, node: "NODE_TYPE", action: "ACTION_TYPE", next_node: "NODE_TYPE"
    ) -> rv_continuous:
        """
        returns the distribution over rewards in the zero one interval when transitioning from a given node and action
        to a given next node. Note that rescaling the rewards to a different range is handled separately.
        """

    @abc.abstractmethod
    def _get_starting_node_sampler(self) -> NextStateSampler:
        """
        returns a sampler over next states that corresponds to the starting state distribution.
        """

    @abc.abstractmethod
    def _check_parameters_in_input(self):
        """
        checks whether the parameters given in input can produce a correct MDP instance.
        """
        assert self._p_rand is None or (0 < self._p_rand < 0.9999)
        assert self._p_lazy is None or (0 < self._p_lazy < 0.9999)

    @abc.abstractmethod
    def get_grid_representation(self, node: "NODE_TYPE", h: int = None):
        """
        produces an ASCII representation of the node given in input.
        """

    @abc.abstractmethod
    def _get_grid_representation(self, node: "NODE_TYPE"):
        """
        produces an ASCII representation of the node given in input.
        """

    @property
    @abc.abstractmethod
    def _possible_starting_nodes(self) -> List["NODE_TYPE"]:
        """
        returns a list containing all the possible starting node for this MDP instance.
        """

    @property
    @abc.abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """
        returns the parameters of the MDP.
        """
        return dict(
            seed=self._seed,
            randomize_actions=self._randomize_actions,
            p_lazy=self._p_lazy,
            p_rand=self._p_rand,
            rewards_range=self._rewards_range,
            make_reward_stochastic=self._make_reward_stochastic,
            variance_multipliers=self._variance_multipliers,
        )

    @property
    def hash(self) -> str:
        """
        returns a hash value based on the parameters of the MDP. This can be use to create cache files.
        """
        s = "_".join(map(str, clean_for_storing(list(self.parameters.values()))))
        return f"mdp_{type(self).__name__}_" + clean_for_file_path(s)

    def __str__(self) -> str:
        """
        returns a string containing all the information about the MDP including parameters, measures of hardness and
        graph metrics.
        """
        string = type(self).__name__ + "\n"
        m_l = 0
        for k, v in self.summary.items():
            m_l = max(m_l, len(max(v.keys(), key=len)) + 4)
        for k, v in self.summary.items():
            string += "\t" + k + "\n"
            for kk, vv in v.items():
                string += f"\t\t{kk}{' ' * (m_l - len(kk))}:\t{vv}\n"
        return string

    @abc.abstractmethod
    def __init__(
        self: Union["EpisodicMDP", "ContinuousMDP"],
        seed: int,
        randomize_actions: bool = True,
        make_reward_stochastic=False,
        variance_multipliers: float = 1.0,
        p_lazy: float = None,
        p_rand: float = None,
        rewards_range: Tuple[float, float] = (0.0, 1.0),
        representation_mapping: Type[RepresentationMapping] = None,
        representation_mapping_kwargs: Dict[str, Any] = dict(),
        hardness_reports_folder="hardness_reports" + os.sep,
        instantiate_mdp: bool = True,
        force_sparse_transition : bool = False
    ):
        """
        instantiates the MDP.

        Parameters
        ----------
        seed : int
            is the random seed.
        randomize_actions : bool, optional
            checks whether to apply a random mapping to the actions for each state. This avoids issues linked to
            the possible bias of the agents to always take action zero at the beginning of the interactions.
            By default, it is set to true.
        variance_multipliers : float, optional
            A constant that can be used to increase the variance of the reward distributions without changing their means.
            The lower the value, the higher the variance. By default, it is set to 1.
        p_lazy : float, optional
            is the probability of an action not producing any effect on the MDP.
            By default, it is set to zero.
        p_rand : float, optional
            is the probability of selecting an action at random instead of the one specified by the agent.
            By default, it is set to zero.
        rewards_range : Tuple[float, float], optional
            is the maximum value of the reward.
            By default, it is set to the zero one interval.
        representation_mapping : RepresentationMapping
            the representation mapping assigned to each state. By default, no representation mapping is used.
        hardness_reports_folder : str, optional
            is the path where the MDP looks for previously cached hardness reports.
        instantiate_mdp : bool
            checks whether to immediately instantiate the MDP.
        """

        # MDP generic variables
        self._seed = seed
        self._randomize_actions = randomize_actions
        self._make_reward_stochastic = make_reward_stochastic
        self._variance_multipliers = variance_multipliers
        if representation_mapping is not None:
            representation_mapping = representation_mapping(
                self, **representation_mapping_kwargs
            )
        self._representation_mapping = representation_mapping
        self._hardness_reports_folder = hardness_reports_folder
        self._force_sparse_transition = force_sparse_transition
        self._p_rand = p_rand if p_rand is None or p_rand > 0.0 else None
        self._p_lazy = p_lazy if p_lazy is None or p_lazy > 0.0 else None
        self.rewards_range = self._rewards_range = (
            rewards_range
            if rewards_range[0] < rewards_range[1]
            else rewards_range[::-1]
        )
        self._are_all_rewards_deterministic = True
        self._are_all_transition_deterministic = True
        self.r_min, self.r_max = self.rewards_range

        # MDP loop variables
        self._hr = None
        self.cur_node = None
        self.last_edge = None
        self.last_starting_node = None
        self.is_reset_necessary = True
        self.current_timestep = 0
        self._rng = np.random.RandomState(seed)
        self._fast_rng = random.Random(seed)

        # Caching variables. Note that we cannot use lru_cache because it prevents from deleting the objects.
        self._cached_rewards = dict()
        self._cached_reward_distributions = dict()
        self._action_mapping = dict()
        self._communication_class = None
        self._recurrent_nodes_set = None
        if not hasattr(self, "_transition_matrix_and_rewards"):
            self._transition_matrix_and_rewards = None
        self._graph_layout = None
        self._graph_metrics = None
        self._summary = None
        self._diameter = None
        self._sum_reciprocals_suboptimality_gaps = None
        self._optimal_value_norm = dict()
        self._optimal_value = None
        self._worst_value = None
        self._random_value = None
        self._optimal_policy = dict()
        self._worst_policy = dict()
        self._otp = None
        self._omc = None
        self._osd = None
        self._oars = None
        self._oar = None
        self._wtp = None
        self._wmc = None
        self._wsd = None
        self._wars = None
        self._war = None
        self._rtp = None
        self._rmc = None
        self._rsd = None
        self._rars = None
        self._rar = None

        if instantiate_mdp:
            self.instantiate_MDP()

    def instantiate_MDP(self):
        # Instantiating the MDP
        self._check_parameters_in_input()
        self._starting_node_sampler = self._get_starting_node_sampler()
        self.starting_nodes = self._starting_node_sampler.next_nodes
        self.G = nx.DiGraph()
        self._instantiate_mdp()
        self.n_states = len(self.G.nodes)

        # Some shortcuts variables
        if not self.is_episodic():
            self._vi, self._pe = (
                discounted_value_iteration,
                discounted_policy_evaluation,
            )
        self.random_policy = (
            np.ones((self.n_states, self.n_actions), dtype=np.float32) / self.n_actions
        )

        # Cache node to index mapping
        mapping = self._rng.rand(self.n_states, self.n_actions).argsort(1)
        self.node_to_index = dict()
        self.index_to_node = dict()
        for i, node in enumerate(self.G.nodes):
            self.node_to_index[node] = i
            self.index_to_node[i] = node

        # Compute the starting state distribution
        self.starting_distribution = np.zeros(self.n_states)
        self.starting_states = []
        for n, p in self._starting_node_sampler.next_nodes_and_probs:
            s = self.node_to_index[n]
            self.starting_distribution[s] = p
            self.starting_states.append(s)
        self.starting_states_and_probs = list(
            zip(self.starting_states, self._starting_node_sampler.probs)
        )

    def _get_action_mapping(self, node: "NODE_TYPE") -> Tuple["ACTION_TYPE", ...]:
        """
        returns the random action mapping for the node given in input.
        """
        if node not in self._action_mapping:
            self._action_mapping[node] = (
                self._rng.rand(self.n_actions).argsort().tolist()
                if self._randomize_actions
                else list(range(self.n_actions))
            )
        return self._action_mapping[node]

    def _inverse_action_mapping(
        self, node: "NODE_TYPE", action: "ACTION_TYPE"
    ) -> "ACTION_TYPE":
        """
        returns the effective action corresponding to the action given in input in the given node. In other words, it
        reverses the random action mapping.
        """
        return self._get_action_mapping(node)[action]

    def _produce_random_seed(self) -> int:
        """
        returns a new random seed that can be used for internal objects.
        """
        return self._fast_rng.randint(0, 10_000)

    def _instantiate_mdp(self):
        """
        recursively instantiate the MDP.
        """
        for sn in self.starting_nodes:
            instantiate_transitions(self, sn)

    @property
    def T(self) -> np.ndarray:
        """
        is an alias for the transition matrix.
        """
        return self.transition_matrix_and_rewards[0]

    @property
    def R(self) -> np.ndarray:
        """
        is an alias for the rewards matrix.
        """
        return self.transition_matrix_and_rewards[1]

    @property
    def recurrent_nodes_set(self) -> Iterable["NODE_TYPE"]:
        """
        returns the recurrent node set.
        """
        if self._recurrent_nodes_set is None:
            self._recurrent_nodes_set = get_recurrent_nodes_set(
                self.communication_class, self.G
            )
        return self._recurrent_nodes_set

    @property
    def communication_class(
        self: Union["EpisodicMDP", "ContinuousMDP"]
    ) -> MDPCommunicationClass:
        """
        returns the communication class.
        """
        if self._communication_class is None:
            self._communication_class = get_communication_class(
                self.T, self.get_episodic_graph(True) if self.is_episodic() else self.G
            )
        return self._communication_class

    def get_optimal_policy(self, stochastic_form: bool) -> np.ndarray:
        """
        returns the optimal policy. It can be either in the stochastic form, so in the form of dirac probabaility
        distribution or as a simple mapping to integer.
        """
        if stochastic_form not in self._optimal_policy:
            self._optimal_policy[stochastic_form] = get_policy_from_q_values(
                self.optimal_value[0], stochastic_form
            )
        return self._optimal_policy[stochastic_form]

    def get_worst_policy(self, stochastic_form) -> np.ndarray:
        """
        returns the worst performing policy. It can be either in the stochastic form, so in the form of dirac probabaility
        distribution or as a simple mapping to integer.
        """
        if stochastic_form not in self._worst_policy:
            self._worst_policy[stochastic_form] = get_policy_from_q_values(
                self._vi(self.T, -self.R)[0], stochastic_form
            )
        return self._worst_policy[stochastic_form]

    @property
    def optimal_value(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        returns the optimal q and state values.
        """
        if self._optimal_value is None:
            self._optimal_value = self._vi(*self.transition_matrix_and_rewards)
        return self._optimal_value

    @property
    def worst_value(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        returns the q and state values for the worst performing policy.
        """
        if self._worst_value is None:
            self._worst_value = self._pe(
                *self.transition_matrix_and_rewards, self.get_worst_policy(True)
            )
        return self._worst_value

    @property
    def random_value(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        returns the q and state values for the randomly acting policy.
        """
        if self._random_value is None:
            self._random_value = self._pe(
                *self.transition_matrix_and_rewards, self.random_policy
            )
        return self._random_value

    @property
    def optimal_transition_probabilities(self) -> np.ndarray:
        """
        returns the transition probabilities corresponding to the optimal policy.
        """
        if self._otp is None:
            T = self.T_cf if self.is_episodic() else self.T
            pi = (
                self.get_optimal_policy_continuous_form(True)
                if self.is_episodic()
                else self.get_optimal_policy(True)
            )
            self._otp = get_transition_probabilities(T, pi)
        return self._otp

    @property
    def worst_transition_probabilities(self) -> np.ndarray:
        """
        returns the transition probabilities corresponding to the worst performing policy.
        """
        if self._wtp is None:
            T = self.T_cf if self.is_episodic() else self.T
            pi = (
                self.get_worst_policy_continuous_form(True)
                if self.is_episodic()
                else self.get_worst_policy(True)
            )
            self._wtp = get_transition_probabilities(T, pi)
        return self._wtp

    @property
    def random_transition_probabilities(self) -> np.ndarray:
        """
        returns the transition probabilities corresponding to the randomly acting policy.
        """
        if self._rtp is None:
            T = self.T_cf if self.is_episodic() else self.T
            pi = self.random_policy_cf if self.is_episodic() else self.random_policy
            self._rtp = get_transition_probabilities(T, pi)
        return self._rtp

    @property
    def optimal_markov_chain(self) -> MarkovChain:
        """
        returns the Markov chain corresponding to the optimal policy.
        """
        if self._omc is None:
            self._omc = get_markov_chain(self.optimal_transition_probabilities)
        return self._omc

    @property
    def worst_markov_chain(self) -> MarkovChain:
        """
        returns the Markov chain corresponding to the worst performing policy.
        """
        if self._wmc is None:
            self._wmc = get_markov_chain(self.worst_transition_probabilities)
        return self._wmc

    @property
    def random_markov_chain(self) -> MarkovChain:
        """
        returns the Markov chain corresponding to the randomly acting policy.
        """
        if self._rmc is None:
            self._rmc = get_markov_chain(self.random_transition_probabilities)
        return self._rmc

    @property
    def optimal_stationary_distribution(self) -> np.array:
        """
        returns the stationary distribution yielded by the optimal policy.
        """
        if self._osd is None:
            self._osd = get_stationary_distribution(
                self.optimal_transition_probabilities,
                self.starting_states_and_probs,
            )
        return self._osd

    @property
    def worst_stationary_distribution(self) -> np.array:
        """
        returns the stationary distribution yielded by the worst performing policy.
        """
        if self._wsd is None:
            self._wsd = get_stationary_distribution(
                self.worst_transition_probabilities,
                self.starting_states_and_probs,
            )
            assert not np.isnan(self._wsd).any()
        return self._wsd

    @property
    def random_stationary_distribution(self) -> np.array:
        """
        returns the stationary distribution yielded by the randomly acting policy.
        """
        if self._rsd is None:
            self._rsd = get_stationary_distribution(
                self.random_transition_probabilities,
                None,
            )
        return self._rsd

    @property
    def optimal_average_rewards(self) -> np.ndarray:
        """
        returns the expected rewards obtained at each state when following the optimal policy.
        """
        if self._oars is None:
            R = self.R_cf if self.is_episodic() else self.R
            pi = (
                self.get_optimal_policy_continuous_form(True)
                if self.is_episodic()
                else self.get_optimal_policy(True)
            )
            self._oars = get_average_rewards(R, pi)
        return self._oars

    @property
    def worst_average_rewards(self) -> np.ndarray:
        """
        returns the expected rewards obtained at each state when following the worst performing policy.
        """
        if self._wars is None:
            R = self.R_cf if self.is_episodic() else self.R
            pi = (
                self.get_worst_policy_continuous_form(True)
                if self.is_episodic()
                else self.get_worst_policy(True)
            )
            self._wars = get_average_rewards(R, pi)
        return self._wars

    @property
    def random_average_rewards(self) -> np.ndarray:
        """
        returns the expected rewards obtained at each state when following the randomly acting policy.
        """
        if self._rars is None:
            R = self.R_cf if self.is_episodic() else self.R
            pi = self.random_policy_cf if self.is_episodic() else self.random_policy
            self._rars = get_average_rewards(R, pi)
        return self._rars

    @property
    def optimal_average_reward(self) -> float:
        """
        returns the expected average reward obtained when following the optimal policy.
        """
        if self._oar is None:
            self._oar = sum(
                self.optimal_stationary_distribution * self.optimal_average_rewards
            )
        return self._oar

    @property
    def worst_average_reward(self) -> float:
        """
        returns the expected average reward obtained when following the worst performing policy.
        """
        if self._war is None:
            self._war = sum(
                self.worst_stationary_distribution * self.worst_average_rewards
            )
        return self._war

    @property
    def random_average_reward(self) -> float:
        """
        returns the expected average reward obtained when following the randomly acting policy.
        """
        if self._rar is None:
            self._rar = sum(
                self.random_stationary_distribution * self.random_average_rewards
            )
        return self._rar

    @property
    def transition_matrix_and_rewards(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        returns the transition probabilities matrix and the reward matrix.
        """
        if self._transition_matrix_and_rewards is None:
            self._transition_matrix_and_rewards = get_transition_matrix_and_rewards(
                self.n_states,
                self.n_actions,
                self.G,
                self.get_info_class,
                self.get_reward_distribution,
                self.node_to_index,
                self._force_sparse_transition
            )
        return self._transition_matrix_and_rewards

    @property
    def graph_layout(self) -> Dict["NODE_TYPE", Tuple[float, float]]:
        """
        returns a graph layout for the MDP. It can be customized by implementing the custom_graph_layout function.
        """
        if self._graph_layout is None:
            self._graph_layout = (
                self.custom_graph_layout()
                if hasattr(self, "custom_graph_layout")
                else nx.nx_agraph.graphviz_layout(self.G)
            )
        return self._graph_layout

    @property
    def graph_metrics(self) -> dict:
        """
        returns a dictionary containing graph metric for the MDP graph.
        """
        if self._graph_metrics is None:
            G = self.get_episodic_graph(True) if self.is_episodic() else self.G
            self._graph_metrics = dict()
            self._graph_metrics["# nodes"] = len(G.nodes)
            self._graph_metrics["# edges"] = len(G.edges)
        return self._graph_metrics

    @property
    def diameter(self: Union["EpisodicMDP", "ContinuousMDP"]) -> float:
        """
        returns the diameter of the MDP.
        """
        if self._diameter is None:
            if self.hardness_report:
                self._diameter = self.hardness_report["MDP measure of hardness"][
                    "diameter"
                ]
            else:
                self._diameter = get_diameter(
                    self.episodic_transition_matrix_and_rewards[0]
                    if self.is_episodic()
                    else self.T,
                    self.is_episodic(),
                )
        return self._diameter

    @property
    def sum_reciprocals_suboptimality_gaps(
        self: Union["EpisodicMDP", "ContinuousMDP"]
    ) -> float:
        """
        returns the sum of the reciprocals of the sub-optimality gaps.
        """
        if self._sum_reciprocals_suboptimality_gaps is None:
            if self.hardness_report:
                self._sum_reciprocals_suboptimality_gaps = self.hardness_report[
                    "MDP measure of hardness"
                ]["suboptimal_gaps"]
            else:
                self._sum_reciprocals_suboptimality_gaps = (
                    get_sum_reciprocals_suboptimality_gaps(
                        *self.optimal_value,
                        self.reachable_states if self.is_episodic() else None,
                    )
                )
        return self._sum_reciprocals_suboptimality_gaps

    def _compute_value_norm(self, discounted: bool) -> float:
        """
        returns the environmental value norm in the undiscounted or undiscounted setting.
        """
        T, R = (self.T_cf, self.R_cf) if self.is_episodic() else (self.T, self.R)
        V = (
            self.optimal_value_continuous_form[1]
            if self.is_episodic()
            else self.optimal_value[1]
        )
        if discounted:
            return calculate_norm_discounted(T, V)
        return calculate_norm_average(
            T, self.optimal_transition_probabilities, self.optimal_average_rewards
        )

    @property
    def discounted_value_norm(self) -> float:
        """
        returns the discounted environmental value norm.
        """
        if True not in self._optimal_value_norm:
            if (
                self._are_all_transition_deterministic
                and self._are_all_rewards_deterministic
            ):
                self._optimal_value_norm[True] = 0.0
            elif self.hardness_report:
                self._optimal_value_norm[True] = self.hardness_report[
                    "MDP measure of hardness"
                ]["value_norm"]
            else:
                self._optimal_value_norm[True] = self._compute_value_norm(True)
        return self._optimal_value_norm[True]

    @property
    def undiscounted_value_norm(self) -> float:
        """
        returns the undiscounted environmental value norm.
        """
        if False not in self._optimal_value_norm:
            self._optimal_value_norm[False] = self._compute_value_norm(False)
        return self._optimal_value_norm[False]

    @property
    def value_norm(self):
        """
        is an alias for the discounted value norm.
        """
        return self.discounted_value_norm

    @property
    def measures_of_hardness(self) -> Dict[str, float]:
        """
        returns a dictionary containing all the measures of hardness availble.
        """
        return dict(
            diameter=self.diameter,
            suboptimal_gaps=self.sum_reciprocals_suboptimality_gaps,
            value_norm=self.value_norm,
        )

    @property
    def summary(self) -> Dict[str, Dict[str, Any]]:
        """
        returns a dictionary with information about the parameters, the measures of hardness and the graph metrics.
        """
        if self._summary is None:
            self._summary = {
                "Parameters": clean_for_storing(self.parameters),
                "Measure of hardness": clean_for_storing(self.measures_of_hardness),
                "Graph metrics": clean_for_storing(self.graph_metrics),
            }
        return self._summary

    @property
    def hardness_report(self) -> Union[Dict, None]:
        """
        looks for a cached hardness report in the folder given in input when the MDP was created. It returns None if it
        is not able to find it.
        """
        if self._hr is None:
            report_file = find_hardness_report_file(self, self._hardness_reports_folder)
            if report_file:
                with open(report_file, "r") as f:
                    report = yaml.load(f, yaml.Loader)
                self._hr = report
            else:
                self._hr = False
        if self._hr:
            return self._hr
        return None

    def get_info_class(self, n: "NODE_TYPE") -> NodeInfoClass:
        """
        returns the container class (NodeInfoClass) associated with node n.
        """
        return self.G.nodes[n]["info_class"]

    def get_transition_distributions(
        self, node: "NODE_TYPE"
    ) -> Dict[int, Union[rv_continuous, NextStateSampler]]:
        """
        returns a dictionary containing the transition distributions for any action at the node given in input.
        """
        return self.get_info_class(node).transition_distributions

    def get_reward_distribution(
        self, node: "NODE_TYPE", action: "ACTION_TYPE", next_node: "NODE_TYPE"
    ):
        """
        returns the reward distribution for transitioning in next_node when selecting action from node.
        """
        if (node, action, next_node) not in self._cached_reward_distributions:
            self._cached_reward_distributions[
                node, action, next_node
            ] = self._get_reward_distribution(
                node, self._inverse_action_mapping(node, action), next_node
            )
        return self._cached_reward_distributions[node, action, next_node]

    def sample_reward(
        self, node: "NODE_TYPE", action: "ACTION_TYPE", next_node: "NODE_TYPE"
    ) -> float:
        """
        returns a sample from the reward distribution when transitioning in next_node when selecting action from node.
        """
        if (node, action, next_node) not in self._cached_rewards or len(
            self._cached_rewards[node, action, next_node]
        ) == 0:
            self._cached_rewards[node, action, next_node] = (
                self.get_reward_distribution(node, action, next_node)
                .rvs(5000, random_state=self._rng)
                .tolist()
            )
        r = self._cached_rewards[node, action, next_node].pop(0)
        return (
            r * (self.rewards_range[1] - self.rewards_range[0]) - self.rewards_range[0]
        )

    def get_measure_from_name(self, measure_name: str) -> float:
        """
        returns the value of the measure given in input.
        """
        if measure_name == "diameter":
            return self.diameter
        elif measure_name in ["value_norm", "environmental_value_norm"]:
            return self.value_norm
        elif measure_name == "suboptimal_gaps":
            return self.sum_reciprocals_suboptimality_gaps
        else:
            raise ValueError(
                f"{measure_name} is not a valid hardness measure name: available ones are "
                + str(self.get_available_hardness_measures())
            )

    def action_spec(self) -> DiscreteArray:
        """
        returns the action spec of the environment.
        """
        return DiscreteArray(self.n_actions, name="action")

    def observation_spec(self) -> Array:
        """
        returns the observation spec of the environment.
        """
        if self._representation_mapping is None:
            return DiscreteArray(self.n_states, name="observation")
        obs = self.get_observation(self.starting_nodes[0], 0)
        return BoundedArray(obs.shape, obs.dtype, -np.inf, np.inf, "observation")

    def get_observation(self, node: "NODE_TYPE", h: int = None):
        """
        returns the representation corresponding to the node given in input. Episodic MDPs also requires the current
        in-episode time step.
        """
        if self._representation_mapping is None:
            return self.node_to_index[self.cur_node]
        return self._representation_mapping.get_observation(node, h)

    def reset(self) -> dm_env.TimeStep:
        """
        resets the environment to a newly sampled starting node.
        """
        self.necessary_reset = False
        self.h = 0
        self.cur_node = self.last_starting_node = self._starting_node_sampler.sample()
        node_info_class = self.get_info_class(self.cur_node)
        node_info_class.update_visitation_counts()
        return dm_env.restart(self.get_observation(self.cur_node, self.h))

    def step(self, action: int, auto_reset=False) -> dm_env.TimeStep:
        """
        takes a step in the MDP for the given action. When auto_reset is set to True then it automatically reset the
        at the end of the episodes.
        """
        if auto_reset and self.necessary_reset:
            return self.reset()
        assert not self.necessary_reset

        # This can be the in episode time step (episodic setting) or the total numuber of time steps (continuous setting)
        self.h += 1

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
        observation = self.get_observation(self.cur_node, self.h)

        # Wrapping the time step in a dm_env.TimeStep
        if self.is_episodic() and self.h >= self.H:
            self.necessary_reset = True
            if self._representation_mapping is None:
                observation = -1
            else:
                observation = np.zeros_like(self.observation_spec().generate_value())
            return dm_env.termination(reward=reward, observation=observation)
        return dm_env.transition(reward=reward, observation=observation)

    def random_step(self, auto_reset=False) -> Tuple[dm_env.TimeStep, int]:
        """
        takes a step with a random action and returns both the next step and the random action. If auto_reset is set to
        True than it automatically resets episodic MDPs.
        """
        action = int(self._rng.randint(self.action_spec().num_values))
        ts = self.step(action, auto_reset)
        return ts, action

    def get_visitation_counts(
        self, state_only=True
    ) -> Dict[Union["NODE_TYPE", Tuple["NODE_TYPE", "ACTION_TYPE"]], int]:
        """
        when state_only is True it returns the visitation counts for the states and when it is False it returns the
        visitation counts for state action pairs.
        """
        if state_only:
            return {
                node: self.get_info_class(node).state_visitation_count
                for node in self.G.nodes
            }
        return {
            (node, a): self.get_info_class(node).actions_visitation_count[a]
            for node in self.G.nodes
            for a in range(self.n_actions)
        }

    def reset_visitation_counts(self):
        """
        resets the visitation counts to zero for all states and state action pairs.
        """
        for node in self.G.nodes:
            self.get_info_class(node).state_visitation_count = 0
            for a in range(self.n_actions):
                self.get_info_class(node).actions_visitation_count[a] = 0

    def get_value_node_labels(
        self: Union["ContinuousMDP", "EpisodicMDP"], V: np.ndarray = None
    ) -> Dict["NODE_TYPE", float]:
        """
        returns a mapping from node to state values. By default, it uses the optimal values.
        """
        if V is None:
            _, V = self.optimal_value
        else:
            if isinstance(self, EpisodicMDP):
                h, d = V.shape
                assert h == self.H and d == self.n_states
            else:
                assert len(V) == self.n_states
        return {
            node: np.round(
                V[0, self.node_to_index[node]]
                if self.is_episodic()
                else V[self.node_to_index[node]],
                2,
            )
            for node in self.G.nodes
        }
