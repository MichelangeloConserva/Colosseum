import io
from time import time
from typing import TYPE_CHECKING, Dict, Iterable, List, Tuple, Union, Set

import numpy as np
import pandas as pd
import toolz
import tqdm
from matplotlib import pyplot as plt
from tqdm import trange
from wrapt_timeout_decorator import timeout

from colosseum import config
from colosseum.config import process_debug_output
from colosseum.emission_maps import Tabular
from colosseum.experiment.indicators import (
    get_episodic_regrets_and_average_reward_at_time_zero,
)
from colosseum.mdp.utils.markov_chain import get_average_reward
from colosseum.utils.acme import InMemoryLogger
from colosseum.utils.acme.base_logger import Logger
from colosseum.utils.formatter import clear_agent_mdp_class_name

if TYPE_CHECKING:
    from colosseum.mdp import ContinuousMDP, EpisodicMDP, BaseMDP
    from colosseum.agent.agents.base import BaseAgent


class MDPLoop:
    """
    The `MDPLoop` is the object in charge of the agent/MDP interactions and the computation of the performance indicators.
    It also provides limited plotting functionalities.
    """

    @staticmethod
    def get_indicators() -> List[str]:
        """
        Returns
        -------
        List[str]
            The code names for the indicators that are computed by the MDPLoop.
        """
        return [
            "cumulative_expected_reward",
            "cumulative_regret",
            "cumulative_reward",
            "normalized_cumulative_expected_reward",
            "normalized_cumulative_regret",
            "normalized_cumulative_reward",
            "steps_per_second",
        ]

    @staticmethod
    def get_baseline_indicators() -> List[str]:
        """
        Returns
        -------
        List[str]
            The code names for the baseline indicators that are computed by the MDPLoop.
        """
        return [
            "random_cumulative_regret",
            "random_cumulative_expected_reward",
            "random_normalized_cumulative_regret",
            "random_normalized_cumulative_expected_reward",
            "optimal_cumulative_expected_reward",
            "optimal_normalized_cumulative_expected_reward",
            "worst_cumulative_regret",
            "worst_cumulative_expected_reward",
            "worst_normalized_cumulative_regret",
            "worst_normalized_cumulative_expected_reward",
        ]

    @staticmethod
    def get_baselines() -> Set[str]:
        """
        Returns
        -------
        Set[str]
            The baselines available for comparison.
        """
        return set(b[: b.find("_")] for b in MDPLoop.get_baseline_indicators())

    @staticmethod
    def get_baselines_color_dict() -> Dict[str, str]:
        """
        Returns
        -------
        Dict[str, str]
            The color associated by default to the baselines.
        """
        return dict(random="black", worst="crimson", optimal="gold")

    @staticmethod
    def get_baselines_style_dict():
        """
        Returns
        -------
        Dict[str, str]
            The line style associated by default to the baselines.
        """
        return dict(random=(0, (6, 12)), worst=(9, (6, 12)), optimal=(0, (6, 12)))

    def __init__(
        self,
        mdp: Union["BaseMDP", "EpisodicMDP", "ContinuousMDP"],
        agent: "BaseAgent",
        logger: Logger = None,
        n_log_intervals_to_check_for_agent_optimality: int = 10,
        enforce_time_constraint: bool = True,
    ) -> object:
        """
        Parameters
        ----------
        mdp: Union["EpisodicMDP", "ContinuousMDP"]
            The MDP.
        agent : BaseAgent
            The agent.
        logger : Logger
            The logger where the results of the interaction between the agent and the MDP are stored. By default, the
            `InMemoryLogger` is used.
        n_log_intervals_to_check_for_agent_optimality : int
            The length of the interval between check is the policy has reached optimality. By default, the check happens
            every ten interactions.
        enforce_time_constraint : bool
            If True, the computational time constraint given in the `run` function is enforced through multithreading.
            By default, it is enforced.
        """

        if logger is None:
            logger = InMemoryLogger()

        self.logger = logger
        self._enforce_time_constraint = enforce_time_constraint
        self._mdp = mdp
        self._agent = agent
        self._episodic = self._mdp.is_episodic()
        self._n_steps_to_check_for_agent_optimality = (
            n_log_intervals_to_check_for_agent_optimality
        )
        assert self._episodic == agent.is_episodic()
        assert self._agent.is_emission_map_accepted(
            Tabular if self._mdp.emission_map is None else self._mdp.emission_map
        )
        self.actions_sequence = []

    @property
    def remaining_time(self) -> float:
        """
        Returns
        -------
        float
            The remaining computational time for training the agent.
        """
        return self._max_time - (time() - self._mdp_loop_timer)

    def _limit_update_time(self, t, f):
        try:
            if self.remaining_time < 0.5:
                raise TimeoutError()
            timeout(self.remaining_time)(f)()
        except TimeoutError or SystemError:
            if config._DEBUG_LEVEL > 0:
                print("Time exceeded with function ", f)
            self._limit_exceeded(t)

    def _limit_exceeded(self, t):
        self._is_training = False
        self._last_training_step = t
        if config._DEBUG_LEVEL > 0:
            do = f"Stopped training at {time() - self._mdp_loop_timer:.2f}"
            process_debug_output(do)
        if self._verbose:
            self._verbose_postfix["is_training"] = f"No, time exhausted at {t}"

    def run(
        self,
        T: int,
        log_every: int = -1,
        max_time: float = np.inf,
    ) -> Tuple[int, Dict[str, float]]:
        """
        runs the agent/MDP interactions.

        Parameters
        ----------
        T : int
            The number of total interactions between the agent and the MDP.
        log_every : int
            The number of time steps after which performance indicators are calculated. By default, it does not calculate
            them at any time except at the last one.
        max_time : float
            The maximum number of seconds the interactions can take. If it is surpassed then the loop is interrupted.
            By default, the maximum given time is infinite.

        Returns
        ----------
        int
            The time step at which the training has been interrupted due to the time constraint. If the constraint has
            been respected it returns -1.
        Dict[str, float]
            The performance indicators computed at the end of the interactions.
        """

        if max_time == np.inf:
            enforce_time_constraint = False
        else:
            enforce_time_constraint = self._enforce_time_constraint

        assert (
            type(log_every) == int
        ), f"The log_every variable should be an integer, received value: {log_every}."
        log_every = -1 if log_every == 0 else log_every

        # Reset the visitation count of the MDP
        self._mdp.reset_visitation_counts()

        self._reset_run_variables()
        self._max_time = max_time

        ts = self._mdp.reset()
        first_before_new_episode_timer = time()
        if enforce_time_constraint and self.remaining_time < np.inf:
            self._limit_update_time(0, self._agent.before_start_interacting)
        else:
            self._agent.before_start_interacting()
        if config._DEBUG_LEVEL > 0:
            if self._is_training:
                do = f"before_start_interacting completed in {time() - first_before_new_episode_timer:.2f}."
            else:
                do = "before_start_interacting exceeded the time limit."
            process_debug_output(do)

        self._set_loop(T)
        for t in self._loop:
            if self._is_training and self.remaining_time < 0.5:
                self._limit_exceeded(t)

            # MDP step
            h = self._mdp.h
            action = self._agent.select_action(ts, h)
            new_ts = self._mdp.step(action)
            self.actions_sequence.append(new_ts.reward)

            # Single step agent update
            if self._is_training:
                if enforce_time_constraint and self.remaining_time < np.inf:
                    self._limit_update_time(
                        t,
                        lambda: self._agent.step_update(ts, action, new_ts, h),
                    )
                else:
                    self._agent.step_update(ts, action, new_ts, h)

            # End of episode agent update
            if self._is_training and self._agent.is_episode_end(ts, action, new_ts, h):
                if enforce_time_constraint and self.remaining_time < np.inf:
                    self._limit_update_time(t, self._agent.episode_end_update)
                else:
                    self._agent.episode_end_update()

            if t > 0 and log_every > 0 and t % log_every == 0:
                # Log the performance of the agent
                self._update_performance_logs(t)
                self._n_steps_since_last_log = 0

                # User defined custom log
                self._agent.agent_logs()

                # Verbose loggings
                self._update_user_loggings(t)

                # Storing the latest regrets
                self._latest_expected_regrets.append(self._normalized_regret)
                if (
                    len(self._latest_expected_regrets)
                    > self._n_steps_to_check_for_agent_optimality
                ):
                    self._latest_expected_regrets.pop(0)

                # Stop training if the agent has confidently reached the optimal policy
                if self._is_training and t > 0.2 * T and self._is_policy_optimal():
                    if type(self._loop) == tqdm.std.tqdm:
                        self._verbose_postfix["is_training"] = f"No, optimal at {t}"
                    self._is_training = False

            self._n_steps_since_last_log += 1
            self._cumulative_reward += new_ts.reward
            ts = new_ts

            # Resetting episodic MDPs
            if self._mdp.is_episodic() and new_ts.last():
                assert self._mdp.necessary_reset or t == T - 2
                ts = self._mdp.reset()
                self._n_episodes += 1

        self._update_performance_logs(t)
        self.logger.close()
        return self._last_training_step, self._last_logs

    def _reset_run_variables(self):
        self._cumulative_reward = 0.0
        self._cumulative_regret = 0.0
        self._normalized_cumulative_regret = 0.0
        self._random_cumulative_expected_reward = 0.0
        self._random_cumulative_regret = 0.0
        self._normalized_random_cumulative_regret = 0.0
        self._cumulative_expected_reward_agent = 0.0

        self._verbose = False
        self._verbose_postfix = dict(is_training="True")
        self._is_training = True
        self._n_steps_since_last_log = 0
        self._last_training_step = -1
        self._n_episodes = 0
        self._last_logs = None
        self._past_logs = None
        self._cached_episodic_regrets = None
        self._cached_continuous_regrets = None
        self._latest_expected_regrets = []

        # Cache the regret for the random agent
        if self._episodic:

            # Random agent regret
            self._episodic_regret_random_agent = (
                self._mdp.episodic_optimal_average_reward
                - self._mdp.episodic_random_average_reward
            )
            self._episodic_normalized_regret_random_agent = (
                self._episodic_regret_random_agent
                / (
                    self._mdp.episodic_optimal_average_reward
                    - self._mdp.episodic_worst_average_reward
                )
            )

            # Worst agent regret
            self._episodic_regret_worst_agent = (
                self._mdp.episodic_optimal_average_reward
                - self._mdp.episodic_worst_average_reward
            )
            self._episodic_normalized_regret_worst_agent = (
                self._episodic_regret_worst_agent
                / (
                    self._mdp.episodic_optimal_average_reward
                    - self._mdp.episodic_worst_average_reward
                )
            )

            # Reward normalized
            self._cumulative_reward_normalizer = lambda t, cr: (
                cr - t * self._mdp.episodic_worst_average_reward
            ) / (
                self._mdp.episodic_optimal_average_reward
                - self._mdp.episodic_worst_average_reward
            )
        else:

            # Random agent regret
            self._regret_random_agent = (
                self._mdp.optimal_average_reward - self._mdp.random_average_reward
            )
            self._normalized_regret_random_agent = self._regret_random_agent / (
                self._mdp.optimal_average_reward - self._mdp.worst_average_reward
            )

            # Worst agent regret
            self._regret_worst_agent = (
                self._mdp.optimal_average_reward - self._mdp.worst_average_reward
            )
            self._normalized_regret_worst_agent = self._regret_worst_agent / (
                self._mdp.optimal_average_reward - self._mdp.worst_average_reward
            )

            assert (
                self._mdp.optimal_average_reward - self._mdp.worst_average_reward
                > 0.0002
            ), type(self._mdp).__name__ + str(self._mdp.parameters)

            self._cumulative_reward_normalizer = lambda t, cr: (
                cr - t * self._mdp.worst_average_reward
            ) / (self._mdp.optimal_average_reward - self._mdp.worst_average_reward)

        self.logger.reset()
        self._mdp_loop_timer = time()
        self._verbose_time = time()

    def _update_performance_logs(self, t: int):
        self._compute_performance_indicators(t + 1)

        self._last_logs = dict(
            steps=t,
            cumulative_regret=self._cumulative_regret,
            cumulative_reward=self._cumulative_reward,
            cumulative_expected_reward=self._cumulative_expected_reward_agent,
            normalized_cumulative_regret=self._normalized_cumulative_regret,
            normalized_cumulative_reward=self._cumulative_reward_normalizer(
                t, self._cumulative_reward
            ),
            normalized_cumulative_expected_reward=self._cumulative_reward_normalizer(
                t, self._cumulative_expected_reward_agent
            ),
            random_cumulative_regret=self._cumulative_regret_random_agent,
            random_cumulative_expected_reward=self._cumulative_reward_random_agent,
            random_normalized_cumulative_regret=self._normalized_cumulative_regret_random_agent,
            random_normalized_cumulative_expected_reward=self._cumulative_reward_normalizer(
                t, self._cumulative_reward_random_agent
            ),
            worst_cumulative_regret=self._cumulative_regret_worst_agent,
            worst_cumulative_expected_reward=self._cumulative_reward_worst_agent,
            worst_normalized_cumulative_regret=self._normalized_cumulative_regret_worst_agent,
            worst_normalized_cumulative_expected_reward=self._cumulative_reward_normalizer(
                t, self._cumulative_reward_worst_agent
            ),
            optimal_cumulative_expected_reward=self._cumulative_reward_optimal_agent,
            optimal_normalized_cumulative_expected_reward=self._cumulative_reward_normalizer(
                t, self._cumulative_reward_optimal_agent
            ),
            steps_per_second=t / (time() - self._mdp_loop_timer),
        )

        # Communicate the indicators to the logger with a maximum of five digits
        a = toolz.valmap(lambda x: np.round(x, 5), self._last_logs)
        self.logger.write(a)

    def _compute_regrets(self):
        if self._episodic:
            return self._compute_episodic_regret()
        return self._compute_continuous_regret()

    def _compute_performance_indicators(self, t: int):
        self._compute_regrets()

        if self._episodic:
            # Randon agent (regret)
            self._cumulative_regret_random_agent = (
                self._episodic_regret_random_agent * t
            )
            self._normalized_cumulative_regret_random_agent = (
                self._episodic_normalized_regret_random_agent * t
            )

            # Worst agent (regret)
            self._cumulative_regret_worst_agent = self._episodic_regret_worst_agent * t
            self._normalized_cumulative_regret_worst_agent = (
                self._episodic_normalized_regret_worst_agent * t
            )

            # Random agent (reward)
            self._cumulative_reward_random_agent = (
                self._mdp.episodic_random_average_reward * t
            )

            # Worst agent (reward)
            self._cumulative_reward_worst_agent = (
                self._mdp.episodic_worst_average_reward * t
            )

            # Optimal agent (reward)
            self._cumulative_reward_optimal_agent = (
                self._mdp.episodic_optimal_average_reward * t
            )

        else:
            # Randon agent (regret)
            self._cumulative_regret_random_agent = self._regret_random_agent * t
            self._normalized_cumulative_regret_random_agent = (
                self._normalized_regret_random_agent * t
            )

            # Worst agent (regret)
            self._cumulative_regret_worst_agent = self._regret_worst_agent * t
            self._normalized_cumulative_regret_worst_agent = (
                self._normalized_regret_worst_agent * t
            )

            # Random agent (reward)
            self._cumulative_reward_random_agent = self._mdp.random_average_reward * t

            # Worst agent (reward)
            self._cumulative_reward_worst_agent = self._mdp.worst_average_reward * t

            # Optimal agent (reward)
            self._cumulative_reward_optimal_agent = self._mdp.optimal_average_reward * t

        # Avoid numerical errors that lead to negative rewards
        assert (
            self._regret >= 0.0
        ), f"{self._regret} on {type(self._mdp).__name__} {self._mdp.parameters} for policy {self._agent.current_optimal_stochastic_policy}"
        assert self._normalized_regret >= 0.0, self._normalized_regret

        self._cumulative_regret += self._regret * self._n_steps_since_last_log
        self._normalized_cumulative_regret += (
            self._normalized_regret * self._n_steps_since_last_log
        )
        self._cumulative_expected_reward_agent += (
            self._agent_average_reward * self._n_steps_since_last_log
        )

    @property
    def _agent_average_reward(self):
        if self._episodic:
            return self._episodic_agent_average_reward / self._mdp.H
        return self._agent_continuous_average_reward

    def _compute_continuous_regret(self):
        if not self._is_training:
            if self._cached_continuous_regrets is None:
                self._cached_continuous_regrets = self._get_continuous_regrets()
            self._regret, self._normalized_regret = self._cached_continuous_regrets
        else:
            self._regret, self._normalized_regret = self._get_continuous_regrets()

    def _get_continuous_regrets(self):
        self._agent_continuous_average_reward = get_average_reward(
            self._mdp.T,
            self._mdp.R,
            self._agent.current_optimal_stochastic_policy,
            [(self._mdp.node_to_index[self._mdp.cur_node], 1.0)],
        )

        r = self._mdp.optimal_average_reward - self._agent_continuous_average_reward
        if np.isclose(r, 0.0, atol=1e-3):
            r = 0.0
        if r < 0:
            r = 0
        nr = r / (self._mdp.optimal_average_reward - self._mdp.worst_average_reward)
        return r, nr

    def _compute_episodic_regret(self):
        if not self._is_training:
            # If the agent is not training, the policy will not change we can cache and reuse the regret for each given
            # starting state.
            if self._cached_episodic_regrets is None:
                Rs, epi_agent_ar = get_episodic_regrets_and_average_reward_at_time_zero(
                    self._mdp.H,
                    self._mdp.T,
                    self._mdp.R,
                    self._agent.current_optimal_stochastic_policy,
                    self._mdp.starting_state_distribution,
                    self._mdp.optimal_value_functions[1],
                )
                self._episodic_agent_average_reward = epi_agent_ar
                self._cached_episodic_regrets = {
                    n: (
                        Rs[self._mdp.node_to_index[n]] / self._mdp.H,  # expected regret
                        Rs[self._mdp.node_to_index[n]]  # normalized expected regret
                        / self._mdp.get_minimal_regret_for_starting_node(n),
                    )
                    for n in self._mdp.starting_nodes
                }
            self._regret, self._normalized_regret = self._cached_episodic_regrets[
                self._mdp.last_starting_node
            ]
        else:
            Rs, epi_agent_ar = get_episodic_regrets_and_average_reward_at_time_zero(
                self._mdp.H,
                self._mdp.T,
                self._mdp.R,
                self._agent.current_optimal_stochastic_policy,
                self._mdp.starting_state_distribution,
                self._mdp.optimal_value_functions[1],
            )
            self._episodic_agent_average_reward = epi_agent_ar
            self._regret = (
                Rs[self._mdp.node_to_index[self._mdp.last_starting_node]] / self._mdp.H
            )
            self._normalized_regret = (
                self._regret
                / self._mdp.get_minimal_regret_for_starting_node(
                    self._mdp.last_starting_node
                )
                * self._mdp.H
            )

    def _is_policy_optimal(self) -> bool:
        if (
            len(self._latest_expected_regrets)
            == self._n_steps_to_check_for_agent_optimality
            and np.isclose(
                0,
                self._latest_expected_regrets,
                atol=1e-4 if self._mdp.is_episodic() else 1e-5,
            ).all()
        ):
            # After we get an empirical suggestions that the policy may be optimal, we check if the expected regret is
            # zero as well
            self._compute_regrets()
            return np.isclose(self._normalized_regret, 0).all()
        return False

    def _set_loop(self, T: int) -> Iterable:
        """
        creates a loop lasting for T steps taking into account the verbosity level.
        """
        if config.VERBOSE_LEVEL != 0:
            desc = f"Experiment loop {type(self._agent).__name__}@{type(self._mdp).__name__}"
            if type(config.VERBOSE_LEVEL) == str:
                self.s = io.StringIO()  # we need this reference
                self._loop = trange(T, desc=desc, file=self.s, mininterval=5)
            else:
                self._loop = trange(T, desc=desc, mininterval=5)
            self._verbose = True
        else:
            self._loop = range(T)

    def _update_user_loggings(self, t: int):
        if self._verbose:  # and time() - self._verbose_time > 5:
            self._verbose_postfix["Instantaneous normalized regret"] = np.round(
                self._normalized_regret / t, 8
            )
            self._loop.set_postfix(self._verbose_postfix, refresh=False)

    def plot(
        self,
        indicator: str = "cumulative_regret",
        ax=None,
        baselines=("random", "worst", "optimal"),
        label=None,
    ):
        """
        plots the values of the indicator obtained by the agent during the interactions along with the baseline values.

        Parameters
        ----------
        indicator : str
            The code name of the performance indicator that will be shown in the plot. Check `MDPLoop.get_indicators()`
            to get a list of the available indicators. By default, the 'cumulative_regret' is shown.
        ax : plt.Axes
            The ax object where the plot will be put. By default, a new axis is created.
        baselines : List[str]
            The baselines to be included in the plot. Check `MDPLoop.get_baselines()` to get a list of the available
            baselines. By default, all baselines are shown.
        label : str
            The label to be given to the agent. By default, a cleaned version of the agent class name is used.
        """

        show = ax is None
        if ax is None:
            fig, ax = plt.subplots()

        assert indicator in self.get_indicators(), (
            f"{indicator} is not an indicator. The indicators available are: "
            + ",".join(self.get_indicators())
            + "."
        )

        df_e = pd.DataFrame(self.logger.data)
        time_steps = [0] + df_e.loc[:, "steps"].tolist()
        ax.plot(
            time_steps[1:] if indicator == "steps_per_second" else time_steps,
            ([] if indicator == "steps_per_second" else [0])
            + df_e.loc[:, indicator].tolist(),
            label=clear_agent_mdp_class_name(type(self._agent).__name__)
            if label is None
            else label,
        )
        ax.set_ylabel(indicator.replace("_", " ").capitalize())

        for b in baselines:
            indicator = indicator.replace(
                "cumulative_reward", "cumulative_expected_reward"
            )
            if b + "_" + indicator in self.get_baseline_indicators():
                ax.plot(
                    time_steps,
                    [0] + df_e.loc[:, b + "_" + indicator].tolist(),
                    label=b.capitalize(),
                    # alpha=0.9,
                    linestyle=(0, (5, 10)),
                    color="darkolivegreen"
                    if "optimal" in b
                    else ("darkred" if "worst" in b else "darkslategray"),
                    linewidth=2,
                )

        ax.set_xlabel("time step")
        ax.legend()
        if show:
            plt.tight_layout()
            plt.show()
