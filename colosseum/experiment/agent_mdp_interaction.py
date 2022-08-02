import io
from time import time
from typing import TYPE_CHECKING, Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
import toolz
import tqdm
from matplotlib import pyplot as plt
from tqdm import trange
from wrapt_timeout_decorator import timeout

from colosseum import config
from colosseum.config import process_debug_output
from colosseum.experiment.utils import get_episodic_regrets_and_average_reward
from colosseum.mdp.utils.markov_chain import get_average_reward
from colosseum.utils.acme import InMemoryLogger
from colosseum.utils.acme.base_logger import Logger

if TYPE_CHECKING:
    from colosseum.mdp import ContinuousMDP, EpisodicMDP
    from colosseum.agent.agents import BaseAgent


class MDPLoop:
    @staticmethod
    def get_available_indicators() -> List[str]:
        """
        returns the code names for the indicators that are calculated by the MDPLoop.
        """
        return [
            "cumulative_reward",
            "cumulative_expected_reward",
            "normalized_cumulative_expected_reward",
            "random_cumulative_reward",
            "normalized_cumulative_reward",
            "random_cumulative_reward",
            "cumulative_regret",
            "random_cumulative_regret",
            "steps_per_second",
            "normalized_cumulative_regret",
            "random_normalized_cumulative_regret",
        ]

    def __init__(
        self,
        mdp: Union["BaseMDP", "EpisodicMDP", "ContinuousMDP"],
        agent: "BaseAgent",
        logger: Logger = InMemoryLogger(),
        n_log_intervals_to_check_for_agent_optimality: int = 10,
        enforce_time_constraint: bool = True,
    ):
        """
        Parameters
        ----------
        mdp: Union["EpisodicMDP", "ContinuousMDP"]
            the MDP.
        agent : BaseAgent
            the agent.
        logger : Logger, optional
            a logger used to store the result of the interaction between the agent and the MDP.
        """
        self.logger = logger
        self._enforce_time_constraint = enforce_time_constraint
        self._mdp = mdp
        self._agent = agent
        self._episodic = self._mdp.is_episodic()
        self._n_steps_to_check_for_agent_optimality = (
            n_log_intervals_to_check_for_agent_optimality
        )
        assert self._episodic == agent.is_episodic()
        self.actions_sequence = []

    @property
    def remaining_time(self) -> float:
        return self._max_time - (time() - self._mdp_loop_timer)

    def _limit_update_time(self, t, f):
        try:
            if self.remaining_time < 0.5:
                raise TimeoutError()
            timeout(self.remaining_time)(f)()
        except TimeoutError or SystemError:
            if config.DEBUG_LEVEL > 0:
                print("Time exceeded with function ", f)
            self._limit_exceeded(t)

    def _limit_exceeded(self, t):
        self._is_training = False
        self._last_training_step = t
        if config.DEBUG_LEVEL > 0:
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

        Parameters
        ----------
        T : int
            number of total interactions between the agent and the MDP.
        log_every : int, optional
            the number of time steps after which performance indicators are calculated. By default, it does not calculate
            them at any time except at the last one.
        max_time : float, optional
            the maximum number of seconds the interactions can take. If it is surpassed then the loop is interrupted.
            By default, the maximum given time is infinite.

        Returns
        ----------
        a tuple containing the time step at which the training has been interrupted due to the time constraint, which is
        -1 if the constraint has been respected, and a dictionary containing the performance indicators.
        """
        if max_time == np.inf:
            enforce_time_constraint = False
        else:
            enforce_time_constraint = self._enforce_time_constraint

        assert (
            type(log_every) == int
        ), f"The log_every variable should be an integer, received value: {log_every}."
        log_every = -1 if log_every == 0 else log_every

        self._reset_run_variables()
        self._max_time = max_time

        ts = self._mdp.reset()
        first_before_new_episode_timer = time()
        if enforce_time_constraint and self.remaining_time < np.inf:
            self._limit_update_time(0, self._agent.before_start_interacting)
        else:
            self._agent.before_start_interacting()
        if config.DEBUG_LEVEL > 0:
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
        self._random_cumulative_reward = 0.0
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
        else:
            self._regret_random_agent = (
                self._mdp.optimal_average_reward - self._mdp.random_average_reward
            )
            self._normalized_regret_random_agent = self._regret_random_agent / (
                self._mdp.optimal_average_reward - self._mdp.worst_average_reward
            )
            assert (
                self._mdp.optimal_average_reward - self._mdp.worst_average_reward
                > 0.0002
            ), type(self._mdp).__name__ + str(self._mdp.parameters)

        self.logger.reset()
        self._mdp_loop_timer = time()
        self._verbose_time = time()

    def _update_performance_logs(self, t: int):
        self._compute_performance_indicators(t + 1)

        self._last_logs = dict(
            steps=t,
            steps_per_second=t / (time() - self._mdp_loop_timer),
            cumulative_reward=self._cumulative_reward,
            cumulative_expected_reward=self._cumulative_expected_reward_agent,
            normalized_cumulative_expected_reward=(
                (self._cumulative_expected_reward_agent - self._mdp.r_min)
                / (self._mdp.r_max - self._mdp.r_min)
            ),
            random_cumulative_reward=self._cumulative_reward_random_agent,
            normalized_cumulative_reward=(
                (self._cumulative_reward - self._mdp.r_min)
                / (self._mdp.r_max - self._mdp.r_min)
            ),
            normalized_random_cumulative_reward=(
                (self._cumulative_reward_random_agent - self._mdp.r_min)
                / (self._mdp.r_max - self._mdp.r_min)
            ),
            cumulative_regret=self._cumulative_regret,
            normalized_cumulative_regret=self._normalized_cumulative_regret,
            random_cumulative_regret=self._cumulative_regret_random_agent,
            random_normalized_cumulative_regret=self._normalized_cumulative_regret_random_agent,
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
            self._cumulative_regret_random_agent = (
                self._episodic_regret_random_agent * t
            )
            self._normalized_cumulative_regret_random_agent = (
                self._episodic_normalized_regret_random_agent * t
            )
            self._cumulative_reward_random_agent = (
                self._mdp.episodic_random_average_reward * t
            )
        else:
            self._cumulative_regret_random_agent = self._regret_random_agent * t
            self._normalized_cumulative_regret_random_agent = (
                self._normalized_regret_random_agent * t
            )
            self._cumulative_reward_random_agent = self._mdp.random_average_reward * t

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
            self._mdp.starting_states_and_probs,
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
            # starting node.
            if self._cached_episodic_regrets is None:
                Rs, epi_agent_ar = get_episodic_regrets_and_average_reward(
                    self._mdp.H,
                    self._mdp.T,
                    self._mdp.R,
                    self._agent.current_optimal_stochastic_policy,
                    self._mdp.starting_distribution,
                    self._mdp.optimal_value[1],
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
            Rs, epi_agent_ar = get_episodic_regrets_and_average_reward(
                self._mdp.H,
                self._mdp.T,
                self._mdp.R,
                self._agent.current_optimal_stochastic_policy,
                self._mdp.starting_distribution,
                self._mdp.optimal_value[1],
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
        """
        checks whether the agent has confidently reached an optimal policy by checking if the latest logged regrets are
        all close to zero.
        """
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

        # if time() - self._verbose_time > 5:
        #     self._verbose_time = time()
        #     if self._verbose:
        #         self._loop.set_postfix_str(f"Episode: {self._n_episodes}")
        #     if type(config.VERBOSE_LEVEL) == str:
        #         if not os.path.isdir(
        #             config.VERBOSE_LEVEL[: config.VERBOSE_LEVEL.rfind(os.sep)]
        #         ):
        #             os.makedirs(
        #                 config.VERBOSE_LEVEL[: config.VERBOSE_LEVEL.rfind(os.sep)],
        #                 exist_ok=True,
        #             )
        #         with open(config.VERBOSE_LEVEL, "a") as f:
        #             f.write(
        #                 datetime.datetime.now().strftime("%H:%M:%S")
        #                 + "  "
        #                 + self.s.getvalue()
        #                 .split("\x1b")[0][2:]
        #                 .replace("\x00", "")
        #                 .replace("\n\r", "")
        #                 + "\n"
        #             )
        #         self.s.truncate(0)

    def plot(
        self,
        y: Union[str, List[str], Tuple[str, ...]] = ("cumulative_regret",),
        ax=None,
        labels: Union[str, List[str], Tuple[str, ...]] = None,
            common_label : str = None
    ):
        """
        quick utility function to plot the performance indicators directly from the MDPLoop.
        """
        show = ax is None
        if ax is None:
            fig, ax = plt.subplots()
        if type(y) == str:
            y = [y]
        if type(labels) == str:
            labels = [labels]
        if type(labels) == list:
            assert len(labels) == len(
                y
            ), "Please make sure that the labels corresponds to the measures."

        df_e = pd.DataFrame(self.logger.data)
        time_steps = [0] + df_e.loc[:, "steps"].tolist()
        for i, yy in enumerate(y):
            ax.plot(
                time_steps[1:] if yy == "steps_per_second" else time_steps,
                ([] if yy == "steps_per_second" else [0]) + df_e.loc[:, yy].tolist(),
                label=(yy.replace("_", " ").capitalize()
                if labels is None or labels[i] is None
                else labels[i]) + ("" if common_label is None or "random" in yy.lower() else f" ({common_label})"),
            )
        ax.set_xlabel("time step")
        ax.legend()
        if show:
            plt.tight_layout()
            plt.show()
