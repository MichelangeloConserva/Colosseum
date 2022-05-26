import os
import warnings
from glob import glob
from time import sleep

import pandas as pd
import seaborn as sns

from colosseum.experiments.utils import (
    group_experiments_by,
    retrieve_experiment_prms,
    retrieve_n_seed,
)
from colosseum.utils.logging import extract_option
from colosseum.utils.miscellanea import ensure_folder

sns.set_theme()


def analyze(w=4, experiment_folder="experiments_done"):
    """
    Interactively analyses the experiments in the given folder, skipping the ones whose names starts with underscore.

    Parameters
    ----------
    w : int, optional
        The results of the experiments are plotted in subplots whose number of columns is w.
        By default, it is set to four.
    experiment_folder : str, optional
        The folder in which the experiments are.
    """

    available_experiments = list(
        sorted(
            filter(
                lambda x: x.split(os.sep)[-1][0] != "_",
                glob(f"{ensure_folder(experiment_folder)}*"),
            )
        )
    )

    if len(available_experiments) == 0:
        return warnings.warn(
            f"It was not possible to find ant experiment in {experiment_folder}."
        )

    while True:
        selected_experiments = extract_option(
            map(lambda x: x.split(os.sep)[-1], available_experiments)
        )
        try:
            selected_experiments = int(selected_experiments)
            assert selected_experiments in range(len(available_experiments))
        except:
            break
        exp_to_show = available_experiments[selected_experiments]
        print(f'You have selected the experiment: "{exp_to_show.split(os.sep)[-1]}"')

        # Loading the experiment files
        logs_folders = glob(f"{ensure_folder(exp_to_show)}logs{os.sep}**")
        if len(logs_folders) == 0:
            return warnings.warn(f"No logging is registered for {exp_to_show}.")
        experiments_prms = list(
            map(
                lambda x: x[1:5],
                retrieve_experiment_prms(
                    exp_to_show, retrieve_n_seed(logs_folders), False, num_steps=None
                )[0],
            )
        )

        opt = extract_option(
            (
                "Show the table with the final results.",
                "Show the table with the final results in LaTex.",
                "Proceed to the plotting functionalities.",
            )
        )
        if opt in [0, 1]:
            from colosseum.experiments.visualisation import experiment_summary

            df = experiment_summary(logs_folders)
            if opt == 0:
                with pd.option_context(
                    "display.max_rows", None, "display.max_columns", None
                ):  # more options can be specified also
                    print(df)
            else:
                print(df.to_latex())

        while True:
            option = input(
                "Please select between the following options or "
                "enter any other character to return to the previous menu."
                "\n0) Group by MDP."
                "\n1) Group by agents.\n"
            )
            try:
                option = int(option)
                assert option in [0, 1]
            except:
                break
            from colosseum.experiments.visualisation import plot_experiment

            by_mdp = option == 0
            experiments_dict = group_experiments_by(experiments_prms, by_mdp)
            plot_experiment(experiments_dict, logs_folders, by_mdp, w)
            sleep(1)
        sleep(1)
