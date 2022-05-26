import datetime
import io
import os
import shutil
import time
import warnings
from time import time
from typing import TYPE_CHECKING, Iterable, List, Tuple, Union, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toolz
from timeout_decorator import timeout_decorator
from tqdm import trange

from colosseum.dp.episodic import policy_evaluation
from colosseum.experiments.hardness_reports import (
    find_hardness_report_file,
    store_hardness_report,
)
from colosseum.utils.acme.base_logger import Logger
from colosseum.utils.acme.csv_logger import CSVLogger
from colosseum.utils.acme.in_memory_logger import InMemoryLogger
from colosseum.utils.miscellanea import ensure_folder

if TYPE_CHECKING:
    from colosseum.agents.base import Agent
    from colosseum.mdps import ContinuousMDP, EpisodicMDP, NodeType


def experiment_label(mc, mp, ac, ap):
    mdp_stamp = f"{mp}{Experiment.SEPARATOR_PRMS}{mc.__name__}"
    return (
        mdp_stamp
        + f"{Experiment.SEPARATOR_MDP_AGENT}"
        + f"{ap}{Experiment.SEPARATOR_PRMS}{ac.__name__}"
    )


class Experiment:
    """
    handles the logging and running of an experiment, which is an agent/MDP interaction along with the measures of
    hardness calculation.
    """

    SEPARATOR_MDP_AGENT = "____"
    SEPARATOR_PRMS = "-"

    def __init__(
        self,
        mdp: Union["EpisodicMDP", "ContinuousMDP"],
        mdp_scope: str,
        agent: "Agent",
        agent_scope: str,
        result_folder: str,
    ):
        self.experiment_label = experiment_label(
            type(mdp), mdp_scope, type(agent), agent_scope
        )
        self.logger = CSVLogger(
            result_folder,
            add_uid=False,
            label=self.experiment_label,
            file_name=f"seed{mdp.seed}_logs",
        )
        self.logger = CSVLogger(
            result_folder,
            add_uid=False,
            label=self.experiment_label,
            file_name=f"seed{mdp.seed}_logs",
        )
        self.seed = mdp.seed

        self._result_folder = result_folder
        self._folder = self.logger._directory
        self._agent = agent
        self._mdp = mdp
        self._loop = MDPLoop(mdp, agent, logger=self.logger)

    def __str__(self):
        return self.experiment_label

    def mdp_hardness_report(self) -> bool:
        report = find_hardness_report_file(self._mdp)
        report_found = report is not None
        if not report_found:
            warnings.warn(
                f"The report for {self.mdp_stamp} has not been found."
                f"It will be created now."
            )
            report = store_hardness_report(self._mdp)
        shutil.copyfile(
            report,
            f"{ensure_folder(self._folder)}seed{self.seed}_report.yml",
        )
        return report_found

    def run(
        self,
        num_steps: int,
        max_time: float,
        verbose=None,
    ) -> int:
        """
        The agent/mdp interactions is run for the given number of steps and within the given max_time. It return the last
        training step, which is lower than num_steps if the agents's learning gets interrupted.
        """
        return self._loop.run(T=num_steps, verbose=verbose, max_time=max_time)


class MDPLoop:
    """
    The MDP loop for the agen/MDP interaction.
    """

    @property
    def available_indicators(self):
        return [
            "cumulative_reward",
            "random_cumulative_reward",
            "cumulative_regret",
            "random_cumulative_regret",
            "steps_per_second",
            "normalized_cumulative_regret",
            "random_normalized_cumulative_regret",
        ]

    def __init__(
        self,
        mdp: Union["EpisodicMDP", "ContinuousMDP"],
        actor: "Agent",
        logger: Logger = InMemoryLogger(),
        force_time_contraint=True,
    ):
        """

        Parameters
        ----------
        mdp: Union["EpisodicMDP", "ContinuousMDP"]
            the MDP.
        actor : Agent
            the agent.
        counter : counting.Counter, optional
            a counter object using to count the passing of time and the ticks.
        logger : Logger, optional
            a logger used to store the result of the interaction between the agent and the MDP.
        """

        # Internalize agent and mdp.
        self.force_time_contraint = force_time_contraint
        self._mdp = mdp
        self._actor = actor
        self.logger = logger
        self._episodic = self._mdp.is_episodic()
        assert self._episodic == actor.is_episodic()

        self.last_log_step = 0
        self.last_cumulative_reward = 0
        self.num_episodes = 0
        self.cumulative_reward = 0.0
        self.cumulative_regret = 0.0
        self.normalized_cumulative_regret = 0.0
        self.random_cumulative_reward = 0.0
        self.random_cumulative_regret = 0.0
        self.normalized_random_cumulative_regret = 0.0
        self.step = 0
        self._verbose = False
        self._last_logs = None
        self._past_logs = None
        self.cached_regret = dict() if mdp.is_episodic() else None

        self.logger.reset()

    def _create_loop(self, T: int, verbose: bool) -> Iterable:
        if verbose:
            desc = f"Experiment loop {type(self._actor).__name__}@{type(self._mdp).__name__}"
            if type(verbose) == bool:
                loop = trange(T, desc=desc, mininterval=5)
            else:
                self.s = io.StringIO()
                loop = trange(T, desc=desc, file=self.s, mininterval=5)
        else:
            loop = range(T)
        return loop

    def verbosity(self, debug_timer: float, loop: Iterable, verbose: bool) -> float:
        if time() - debug_timer > 5:
            debug_timer = time()
            loop.set_postfix_str(f"Episode: {self.num_episodes}")
            if type(verbose) == str:
                if not os.path.isdir(verbose[: verbose.rfind(os.sep)]):
                    os.makedirs(verbose[: verbose.rfind(os.sep)], exist_ok=True)
                with open(verbose, "a") as f:
                    f.write(
                        datetime.datetime.now().strftime("%H:%M:%S")
                        + "  "
                        + self.s.getvalue()
                        .split("\x1b")[0][2:]
                        .replace("\x00", "")
                        .replace("\n\r", "")
                        + "\n"
                    )
                self.s.truncate(0)
        return debug_timer

    def plot(self, y: Union[str, List[str]] = ("cumulative_regret",), ax=None):
        show = ax is None
        if ax is None:
            fig, ax = plt.subplots()
        if type(y) == str:
            y = [y]

        df_e = pd.DataFrame(self.logger.data)
        time_steps = [0] + df_e.loc[:, "steps"].tolist()
        for yy in y:
            ax.plot(
                time_steps[1:] if yy == "steps_per_second" else time_steps,
                ([] if yy == "steps_per_second" else [0]) + df_e.loc[:, yy].tolist(),
                label=yy.replace("_", " ").capitalize(),
            )
        ax.set_xlabel("time step")
        ax.legend()
        if show:
            plt.tight_layout()
            plt.show()

    def run(
        self,
        T: int,
        verbose: Union[bool, str] = None,
        log_every: int = None,
        max_time: float = 60 * 60,
    ) -> int:
        """

        Parameters
        ----------
        T : int
            number of total interactions between the agent and the MDP.
        verbose : Union[bool, str]
            when it is set to bool it prints directly on the console and when it is set to a string it saves the outputs
            in a file with such name.
        log_every : int, optional
            the number of time steps after which results are stored.
        max_time : float, optional
            the maximum number of seconds the interactions can take. If surpassed then the loop is interruped.
        """

        self._verbose = verbose
        log_every = (
            max(1000 if T > 5000 else 50, int(T / 50))
            if log_every is None
            else log_every
        )
        if log_every == 0:
            log_every = 5

        loop = self._create_loop(T, verbose)
        ts = self._mdp.reset()

        tt = time()
        if type(self._mdp.verbose) == str:
            with open(self._mdp.verbose, "a") as f:
                f.write("Before first before_new_episode\n")
        self._actor.before_new_episode()
        if type(self._mdp.verbose) == str:
            with open(self._mdp.verbose, "a") as f:
                f.write(f"After first before_new_episode ({time() - tt:.2f})\n")

        self._start = debug_timer = time()
        self.past_regrets = {0: 0}
        self.exp_regrets = dict()
        self.train = True
        last_training_step = -1
        for t in loop:

            # Breaking if time has exceeded
            time_passed = time() - self._start
            if self.train and time_passed > max_time:
                last_training_step = t + 1
                self.train = False
                if type(self._mdp.verbose) == str:
                    with open(self._mdp.verbose, "a") as f:
                        f.write(
                            f"Stopped self.training after {time() - self._start:.2f}\n"
                        )
                if verbose:
                    loop.set_postfix_str("Not training because of timeout.")

            # Acting with the MDP
            action = self._actor.select_action(ts.observation)
            new_ts = self._mdp.step(action)

            # Observing if the agent is still self.training
            if self.train:
                time_passed = time() - self._start
                if self.force_time_contraint:
                    self._actor.ob = timeout_decorator.timeout(
                        max(1.0, max_time - time_passed)
                    )(self._actor.observe)
                else:
                    self._actor.ob = self._actor.observe
                try:
                    self._actor.ob(ts, action, new_ts)
                except:
                    pass

            # End of episode agent update
            if self.train:
                if self._actor.is_episode_end(ts, action, new_ts):
                    time_passed = time() - self._start
                    if self.force_time_contraint:
                        self._actor.bne = timeout_decorator.timeout(
                            max(1.0, max_time - time_passed)
                        )(self._actor.before_new_episode)
                    else:
                        self._actor.bne = self._actor.before_new_episode
                    try:
                        self._actor.bne()
                    except:
                        pass
                    self.num_episodes += 1

            # Logging and checking if the agent has reached the optimal policy
            if t > 0 and t % log_every == 0:
                self.update_logs(t)
                self.last_log_step = self.step
                self.last_cumulative_reward = self.cumulative_reward
                self.exp_regrets[t] = self.r
                if t > 0.2 * T and all(
                    np.isclose(0, x, atol=1e-2 if self._mdp.is_episodic() else 1e-5)
                    for x in list(self.exp_regrets.values())[-10:]
                ):
                    self.train = False
                    if verbose:
                        loop.set_postfix_str(
                            "Not training because optimal policy has been reached."
                        )
                self._actor.debug_info()

            if verbose:
                debug_timer = self.verbosity(debug_timer, loop, verbose)

            # Resetting episodic MDPs
            if self._mdp.is_episodic() and new_ts.last():
                assert self._mdp.necessary_reset or t == T - 2
                ts = self._mdp.reset()
                self.step += 1
                continue

            self.cumulative_reward += new_ts.reward
            self.step += 1
            ts = new_ts

        return last_training_step

    def close(self):
        """
        Terminates the interaction between the agent/MDP interaction and closes the logger.
        """
        self.logger.close()

    def update_logs(self, t: int):
        cr, norm_cr = self.get_cumulative_regret()
        rand_cr, rand_norm_cr = self.cumulative_regret_random()

        assert cr > 0
        assert norm_cr > 0

        self._last_logs = dict(
            steps=self.step,
            cumulative_reward=self.cumulative_reward,
            random_cumulative_reward=self.cumulative_reward_random(),
            cumulative_regret=cr,
            random_cumulative_regret=rand_cr,
            steps_per_second=t / (time() - self._start),
            normalized_cumulative_regret=norm_cr,
            random_normalized_cumulative_regret=rand_norm_cr,
        )
        self.logger.write(toolz.valmap(lambda x: np.round(x, 5), self._last_logs))

    def expected_agent_episode_return(self, n: "NodeType") -> float:
        T, R = self._mdp.transition_matrix_and_rewards
        _, V = policy_evaluation(
            self._mdp.H, T, R, self._actor.current_policy.pi_matrix
        )
        return V[0, self._mdp.node_to_index(n)]

    def get_cumulative_regret(self) -> Tuple[float, float]:
        if self._mdp.is_episodic():
            n = self._mdp.last_starting_node

            if not self.train:
                if n in self.cached_regret:
                    self.r, nr = self.cached_regret[n]
                    self.cumulative_regret += self.r * (self.step - self.last_log_step)
                    self.normalized_cumulative_regret += nr * (
                        self.step - self.last_log_step
                    )
                    return self.cumulative_regret, self.normalized_cumulative_regret

            opt_value = self._mdp.optimal_policy_starting_value(n)
            self.r = opt_value - self.expected_agent_episode_return(n)
            self.nr = self.r / (opt_value - self._mdp.worst_policy_starting_value(n))
            if not self.train:
                self.cached_regret[n] = self.r, self.r / (
                    opt_value - self._mdp.worst_policy_starting_value(n)
                )

            self.cumulative_regret += self.r * (self.step - self.last_log_step)
            self.normalized_cumulative_regret += self.nr * (
                self.step - self.last_log_step
            )
            return self.cumulative_regret, self.normalized_cumulative_regret

        last_steps = self.step - self.last_log_step
        last_rewards = self.cumulative_reward - self.last_cumulative_reward
        self.r = max(
            (0, (self._mdp.optimal_average_reward() - last_rewards / last_steps))
        )
        self.cumulative_regret += self.r * last_steps
        self.normalized_cumulative_regret += (
            self.r
            / (self._mdp.optimal_average_reward() - self._mdp.worst_average_reward())
            * last_steps
        )

        return self.cumulative_regret, self.normalized_cumulative_regret

    def cumulative_regret_random(self) -> Tuple[float, float]:
        if self._mdp.is_episodic():
            n = self._mdp.last_starting_node
            r = self._mdp.optimal_policy_starting_value(
                n
            ) - self._mdp.random_policy_starting_value(n)
            self.random_cumulative_regret += r * (self.step - self.last_log_step)
            self.normalized_random_cumulative_regret += (
                r
                / (
                    self._mdp.optimal_policy_starting_value(n)
                    - self._mdp.worst_policy_starting_value(n)
                )
            ) * (self.step - self.last_log_step)
            return (
                self.random_cumulative_regret,
                self.normalized_random_cumulative_regret,
            )

        return self.step * (
            self._mdp.optimal_average_reward() - self._mdp.random_average_reward()
        ), self.step * (
            (self._mdp.optimal_average_reward() - self._mdp.random_average_reward())
            / (self._mdp.optimal_average_reward() - self._mdp.worst_average_reward())
        )

    def cumulative_reward_random(self) -> float:
        if self._mdp.is_episodic():
            n = self._mdp.last_starting_node
            r = self._mdp.random_policy_starting_value(n)
            self.random_cumulative_reward += r
            return self.random_cumulative_reward
        return self.step * self._mdp.random_average_reward()


def run_experiment_single_thread(*args, verbose=True, **kwargs):
    return run_experiment(*args, **kwargs, force_single_thread=True, verbose=verbose)


def run_experiment(
    num_steps : int,
    seed : int,
    mdp_class : Union[Type["ContinuousMDP"], Type["EpisodicMDP"]],
    mdp_scope : str,
    agent_class : Type["Agent"],
    agent_scope : str,
    result_folder : str,
    gin_config_files_paths : List[str],
    max_time: float,
    force_single_thread : bool =False,
    verbose: Union[str, bool] = "temp_multiprocess",
):
    """

    Parameters
    ----------
    num_steps : int.
        the number of time steps for the agent/MDP interaction.
    seed : int.
        the seed for the agent and the MDP.
    mdp_class : Union[Type["ContinuousMDP"], Type["EpisodicMDP"]].
        the class of the MDP.
    mdp_scope : str.
        the gin scope of the MDP class that defined its parameters. For example, it can be "prms_0".
    agent_class : Type["Agent"].
        the class of the agent.
    agent_scope : str.
        the gin scope of the agent class that defined its parameters. For example, it can be "prms_0".
    result_folder : str.
        the folder in which the logs are stored.
    gin_config_files_paths : List[str].
        the file paths of the gin configuration files.
    max_time : float.
        the maximum training time for the agent.
    force_single_thread : bool, optional.
        whether to enforce the experiment to be run on a single thread rather multiprocessing.
    verbose : Union[str, bool]
        if True, verbose logs will be printed to the console. If given as a string, it will interpret it as the path of
        a folder where to verbose logs will be stored.

    """


    import gin

    from colosseum.experiments.utils import apply_gin_config
    from colosseum.utils.acme.specs import make_environment_spec

    apply_gin_config(gin_config_files_paths)

    if type(verbose) == str:
        verbose = (
            f"{ensure_folder(verbose)}"
            f"{result_folder.replace(os.sep, '_')}_"
            f"{seed}_"
            f"{mdp_scope}{Experiment.SEPARATOR_PRMS}{mdp_class.__name__}_"
            f"{agent_scope}{Experiment.SEPARATOR_PRMS}{agent_class.__name__}.txt"
        )

    with gin.config_scope(mdp_scope):
        mdp = mdp_class(
            seed=seed, force_single_thread=force_single_thread, verbose=verbose
        )
    with gin.config_scope(agent_scope):
        if mdp.is_episodic():
            agent = agent_class(
                environment_spec=make_environment_spec(mdp),
                seed=seed,
                H=mdp.H,
                r_max=mdp.r_max,
                T=num_steps,
            )
        else:
            agent = agent_class(
                environment_spec=make_environment_spec(mdp),
                seed=seed,
                r_max=mdp.r_max,
                T=num_steps,
            )
    exp = Experiment(
        mdp,
        mdp_scope,
        agent,
        agent_scope,
        result_folder,
    )

    if type(verbose) == str:
        with open(verbose, "w") as f:
            f.write(
                f"{seed} {mdp_class} {mdp_scope} {agent_class} {agent_scope} {result_folder}\n"
            )

    start = time.time()
    report_found = exp.mdp_hardness_report()
    if type(verbose) == str:
        with open(verbose, "a") as f:
            f.write(
                f"Hardness report { 'found' if report_found else 'created'} after {time.time() - start:.2f}s\n"
            )
    last_training_step = exp.run(
        num_steps=num_steps, verbose=verbose, max_time=max_time
    )

    if last_training_step != -1:
        with open(f"{ensure_folder(exp._folder)}time_exceeded.txt", "a") as f:
            f.write(
                f"last training step at ({last_training_step}) for {exp.logger.file_path}\n"
            )

    if type(verbose) == str:
        os.remove(verbose)
