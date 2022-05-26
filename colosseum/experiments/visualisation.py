import os
import re
from glob import glob
from typing import List

import matplotlib.colors
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from colosseum.experiments.utils import (
    c,
    get_log_folder,
    get_measure_value,
    retrieve_available_measures,
    retrieve_measure_value,
    unite_seed,
)
from colosseum.utils.logging import extract_option
from colosseum.utils.miscellanea import clear_th, normalize, ensure_folder

no_y_opt = {
    "episode_steps",
    "seed",
    "index",
    "steps",
    "random_cumulative_regret",
    "random_cumulative_reward",
    "random_episode_regret",
    "random_episode_return",
    "random_normalized_cumulative_regret",
}


def create_fig_axes(n, w=4, share_axis=False):
    if n < w:
        w = n
    h = int(np.ceil(n / w))
    ww = max(4, w)
    fig, axes = plt.subplots(
        h, w, figsize=(ww * ww, h * ww), sharey=share_axis, sharex=True
    )
    return fig, axes.ravel()


def experiment_summary2(logs_folders: List[str]):
    from colosseum.experiments.experiment import Experiment

    agents = list(
        sorted(
            list(
                set(
                    [
                        x[
                            x.rfind(Experiment.SEPARATOR_PRMS)
                            + len(Experiment.SEPARATOR_PRMS) :
                        ]
                        for x in logs_folders
                    ]
                )
            )
        )
    )[::-1]

    df = pd.DataFrame(columns=["MDP", "MDP param", *agents])
    df2 = pd.DataFrame(columns=["MDP", "MDP param", *agents])
    for logs_folder in logs_folders:
        (mdp_prm, mdp_class_name), (agent_prm, agent_class_name) = [
            x.split(Experiment.SEPARATOR_PRMS)
            for x in logs_folder[logs_folder.rfind(os.sep) + 1 :].split(
                Experiment.SEPARATOR_MDP_AGENT
            )
        ]
        res_df = unite_seed(logs_folder)
        total_n_steps = np.min(res_df.steps.values[np.nonzero(res_df.steps.values)]) * (
            1 + len(res_df.groupby("steps").count())
        )

        last_steps = []
        for seed in res_df.seed.unique():
            last_steps.append(res_df[res_df.seed.values == seed].steps.tolist()[-1])
        last_regrets = []
        for seed in res_df.seed.unique():
            last_regrets.append(
                res_df[
                    res_df.seed.values == seed
                ].normalized_cumulative_regret.tolist()[-1]
                / total_n_steps
            )

        df.loc[
            mdp_class_name + mdp_prm,
            ["MDP", "MDP param", agents[agents.index(agent_class_name)]],
        ] = [
            mdp_class_name,
            mdp_prm,
            f"${np.mean(last_regrets):.2f}\\pm{np.std(last_regrets):.2f}$",
        ]
        df2.loc[
            mdp_class_name + mdp_prm,
            ["MDP", "MDP param", agents[agents.index(agent_class_name)]],
        ] = [
            mdp_class_name,
            mdp_prm,
            f"${np.mean(last_regrets):.7f}\\pm{np.std(last_regrets):.7f}$",
        ]

    for i in range(len(df.index)):
        m = (
            np.argmin(
                [re.findall("\d+\.\d+", s)[0] for s in df2.iloc[i].values.tolist()[2:]]
            )
            + 2
        )
        df.iloc[i, m] = (
            "$\\textbf{"
            + df.iloc[i, m][1 : df.iloc[i, m].find("\\")]
            + "}"
            + df.iloc[i, m][df.iloc[i, m].find("\\") :]
        )

    for a in agents:
        x = np.vstack(
            df2.applymap(
                lambda s: list(map(float, re.findall("\d+\.\d+", s)))
                if len(re.findall("\d+\.\d+", s)) > 0
                else ""
            )
            .loc[:, a]
            .values
        ).mean(0)

        df.loc[mdp_class_name + mdp_prm + "_", ["MDP", "MDP param", a]] = [
            "Average",
            mdp_prm,
            f"${x[0]:.2f}\\pm{x[1]:.2f}$",
        ]

    return df.set_index(["MDP", "MDP param"]).sort_index()


def experiment_summary(logs_folders: List[str]):
    from colosseum.experiments.experiment import Experiment

    df = pd.DataFrame(
        columns=[
            "MDP",
            "MDP param",
            "Agent",
            "Agent param",
            # "Steps",
            "Normalized cumulative regret",
            "Completed in time",
        ]
    )
    for logs_folder in logs_folders:
        (mdp_prm, mdp_class_name), (agent_prm, agent_class_name) = [
            x.split(Experiment.SEPARATOR_PRMS)
            for x in logs_folder[logs_folder.rfind(os.sep) + 1 :].split(
                Experiment.SEPARATOR_MDP_AGENT
            )
        ]
        res_df = unite_seed(logs_folder)

        last_steps = []
        for seed in res_df.seed.unique():
            last_steps.append(res_df[res_df.seed.values == seed].steps.tolist()[-1])
        last_regrets = []
        for seed in res_df.seed.unique():
            last_regrets.append(
                res_df[
                    res_df.seed.values == seed
                ].normalized_cumulative_regret.tolist()[-1]
            )

        n_failed = 0
        if os.path.exists(ensure_folder(logs_folder) + "time_exceeded.txt"):
            with open(ensure_folder(logs_folder) + "time_exceeded.txt", "r") as f:
                n_failed = len(set(f.readlines()))

        df.loc[len(df)] = [
            mdp_class_name,
            mdp_prm,
            agent_class_name,
            agent_prm,
            f"{np.mean(last_regrets):.2f}\\pm{np.std(last_regrets):.2f}",
            f"{len(last_steps) - n_failed}/{len(last_steps)}",
        ]
    return df.set_index(
        [
            "MDP",
            "MDP param",
            "Agent",
            "Agent param",
        ]
    ).sort_index()


def _plot_results_benchmark(df, y: str, ax, plot_random, label, color: str):
    sns.lineplot(
        x="steps",
        y=y,
        color=color,
        data=df.loc[:, ["steps", y, "seed"]],
        label=clear_th(label[:-1]),
        ci="sd",
        ax=ax,
    )
    if ("regret" in y or "return" in y) and plot_random:
        if "random_" + y in df.columns:
            sns.lineplot(
                x="steps",
                y="random_" + y,
                color=list(matplotlib.colors.TABLEAU_COLORS.keys())[0],
                data=df.loc[:, ["steps", "random_" + y, "seed"]],
                label="Random",
                ci="sd",
                ax=ax,
            )
        if "normalized" in y:
            ax.plot(
                [0, df.steps.tolist()[-1]],
                [0, df.steps.tolist()[-1]],
                color="tab:grey",
                linestyle="--",
            )

    ax.set_ylabel(y.replace("_", " ").capitalize(), fontdict=dict(size=20))
    ax.set_xlabel("time step", fontdict=dict(size=20))
    ax.tick_params(labelsize=18)


def _experiment_plot(
    y, by_mdp, grouped, name, prms, logs_folders, color_dict, selected_measure, ax
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    plot_random = by_mdp
    for i, (cls_inn, prms_inn) in enumerate(grouped):
        inner_name = cls_inn.__name__

        cur_logs_folder = get_log_folder(
            *(
                (name, prms, inner_name, prms_inn)
                if by_mdp
                else (inner_name, prms_inn, name, prms)
            ),
            logs_folders,
        )
        cur_logs_folder = ensure_folder(cur_logs_folder)
        time_steps_failed = dict()
        if os.path.exists(cur_logs_folder + "time_exceeded.txt"):
            with open(cur_logs_folder + "time_exceeded.txt", "r") as f:
                failed = set(f.readlines())
            time_steps_failed = {
                int(fail[fail.find("seed") + 4 : fail.find("_log")]): int(
                    fail[fail.find("(") + 1 : fail.find(")")]
                )
                for fail in failed
            }
        n_seeds = len(glob(cur_logs_folder + "seed*.csv"))

        df = unite_seed(cur_logs_folder)

        assert (
            df is not None
            and (
                df.groupby("seed").count().steps
                == df.groupby("seed").count().steps.min()
            ).all()
        ), f"There is a formatting error with results from {cur_logs_folder}"

        if y is None:
            y_options = list(sorted(set(df.columns) - no_y_opt))
            option = extract_option(y_options)
            if option != -1:
                y = y_options[option]
            else:
                plt.close()
                return

        if by_mdp:
            label = inner_name + prms_inn.split("_")[1]
            if inner_name in color_dict:
                color = color_dict[inner_name]
            else:
                color = list(matplotlib.colors.TABLEAU_COLORS.keys())[
                    len(color_dict) + 1
                ]
                color_dict[inner_name] = color
        else:
            if selected_measure == 0:
                label = inner_name + prms_inn.split("_")[1]
            else:
                label = f"{selected_measure.capitalize()}: {get_measure_value(selected_measure, cur_logs_folder):.2f}"
            color = matplotlib.cm.get_cmap("YlOrRd")(i / len(grouped))

        if len(time_steps_failed) > 0:
            tt = yy = 0
            for s, tf in time_steps_failed.items():
                ttt = df.loc[:, "steps"].tolist()[
                    np.argmin(np.abs(df.loc[:, "steps"] - tf))
                ]
                tt += ttt
                yy += (
                    df[(df.loc[:, "seed"] == s) & (df.loc[:, "steps"] == ttt)]
                    .loc[:, y]
                    .tolist()[0]
                )
            ttt = df.loc[:, "steps"].tolist()[
                np.argmin(np.abs(df.loc[:, "steps"] - tt / n_seeds))
            ]
            ax.text(
                tt / n_seeds,
                df[df.loc[:, "steps"] == ttt].loc[:, y].mean(),
                "}",
                fontdict=dict(size=27),
                color=color,
                verticalalignment="center",
            )
        _plot_results_benchmark(df, y, ax, plot_random, label, color)
        plot_random = False
    ax.set_title(f"{name} ({int(prms.split('_')[1])+1})", fontsize=20)
    ax.legend(fontsize=8)

    return y


def plot_experiment(
    experiments_dict, logs_folders, by_mdp, w=4, max_measures=7, seed: int = None
):
    if seed:
        np.random.seed(seed)

    y = None
    selected_measure = None
    fig, axes = create_fig_axes(len(experiments_dict), w=w)
    color_dict = dict()
    for j, ((cls, prms), grouped) in enumerate(experiments_dict):
        name = cls.__name__

        if not by_mdp and selected_measure is None:
            measures = retrieve_available_measures(logs_folders)
            option = extract_option(["MDP name"] + list(measures))
            if option != -1:
                if option == 0:
                    selected_measure = 0
                else:
                    selected_measure = measures[option - 1]
            else:
                plt.close()
                return
        if selected_measure is not None and selected_measure != 0:
            grouped = sorted(
                list(grouped),
                key=lambda x: -retrieve_measure_value(
                    selected_measure, *x, cls, prms, logs_folders
                ),
            )

            if not by_mdp and len(grouped) > 5:
                grouped = [
                    grouped[x]
                    for x in sorted(
                        np.random.choice(range(len(grouped)), max_measures, False),
                        reverse=True,
                    )
                ]

        y = _experiment_plot(
            y,
            by_mdp,
            grouped,
            name,
            prms,
            logs_folders,
            color_dict,
            selected_measure,
            axes[j],
        )

    for i in range(j + 1, len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()

    os.makedirs("tmp", exist_ok=True)
    name = logs_folders[0]
    f_name = (
        "tmp"
        + os.sep
        + name[name.find(os.sep) + 1 : name.find("logs") - 1]
        + f"_bymdp{by_mdp}"
        + f"_y{y}"
        + (f"_{selected_measure}" if selected_measure is not None else "")
        + "_plot.pdf"
    )
    plt.savefig(f_name, bbox_inches="tight")
    plt.show()


def plot_results_hardness_analysis(
    mdp,
    p_rand,
    n_seeds,
    diameter_values,
    valuenorm_values,
    gaps_values,
    cum_reg_values,
    x,
    x_label,
    approximate_regret,
    save_folder=None,
    save_fig=None,
    optimal_approximation_regret=False,
):
    if not np.isclose(np.ptp(valuenorm_values), 0, atol=1e-3):
        valuenorm_values = normalize(valuenorm_values)
    else:
        valuenorm_values[...] = 0.5
    diameter_values = normalize(diameter_values)
    gaps_values = normalize(gaps_values)
    if approximate_regret:
        cum_reg_values = normalize(cum_reg_values)

    df = pd.DataFrame(
        np.vstack(
            [
                np.hstack(
                    (
                        x.reshape(-1, 1),
                        diameter_values[:, seed : seed + 1],
                        valuenorm_values[:, seed : seed + 1],
                        gaps_values[:, seed : seed + 1],
                        cum_reg_values[:, seed : seed + 1],
                        seed * np.ones((len(x), 1)),
                    )
                )
                for seed in range(n_seeds)
            ]
        ),
        columns=[
            x_label,
            "Diameter",
            "Env-value norm",
            "Sum  of  the  reciprocals \nof  the  sub-optimality  gaps",
            "Cumulative regret",
            "seed",
        ],
    )

    sns.lineplot(
        x=x_label,
        y="Diameter",
        data=df,
        label="Diameter",
        ci="sd",
    )
    sns.lineplot(
        x=x_label,
        y="Env-value norm",
        data=df,
        label="Env-value norm",
        ci="sd",
    )
    sns.lineplot(
        x=x_label,
        y="Sum  of  the  reciprocals \nof  the  sub-optimality  gaps",
        data=df,
        label="Sum  of  the  reciprocals \nof  the  sub-optimality  gaps",
        ci="sd",
    )
    if approximate_regret:
        sns.lineplot(
            x=x_label,
            y="Cumulative regret",
            data=df,
            label="Cumulative regret of\ntuned near-optimal agent.",
            ci="sd",
        )
    plt.ylabel("normalized value")
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.tight_layout()
    if save_folder is not None:
        plt.savefig(
            f"{save_folder}{type(mdp).__name__}_"
            f"{'epi' if mdp.is_episodic() else 'cont'}"
            f"{'_size' if x_label == '# states' else ''}"
            f"{'_lazy' if 'lazy' in x_label else ''}"
            f"{'_prand' if 'random' in x_label else ''}"
            f"{f'rand{c(p_rand)}_' if p_rand is not None else ''}.pdf"
        )
    if save_fig is not None:
        plt.savefig(save_fig)
    plt.show()
