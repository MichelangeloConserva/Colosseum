import re
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from colosseum.analysis.utils import format_indicator_name, get_n_failed_interactions
from colosseum.analysis.utils import get_available_mdps_agents_prms_and_names
from colosseum.analysis.utils import get_formatted_name, get_logs_data
from colosseum.experiment.agent_mdp_interaction import MDPLoop
from colosseum.utils.formatter import clear_agent_mdp_class_name


def get_latex_table_of_average_indicator(
    experiment_folder: str,
    indicator: str,
    show_prm: bool = False,
    divide_by_total_number_of_time_steps: bool = True,
    mdps_on_row: bool = True,
    print_table: bool = False,
    return_table: bool = False,
) -> Union[str, Tuple[str, pd.DataFrame]]:
    r"""
    produces a latex table whose entries are the averages over the seeds of the indicator given in input for the
    results in the experiment folder.

    Parameters
    ----------
    experiment_folder : str
        The folder that contains the experiment logs, MDP configurations and agent configurations.
    indicator : str
        The code name of the performance indicator that will be shown in the plot. Check `MDPLoop.get_indicators()` to
        get a list of the available indicators.
    show_prm : bool
        If True, the gin parameter config index is shown next to the agent/MDP class name. By default, it is not shown.
    divide_by_total_number_of_time_steps : bool
        If True, the value of the indicator is divided by the total number of time steps of agent/MDP interaction. By
        default, it is divided.
    mdps_on_row : bool
        If True, MDPs are shown in the rows. If False, agents are shown on the row indices. By default, MDPs are shown
        on row indices.
    print_table : bool
        If True, the table is printed.
    return_table : bool
        If True, in addition to the string with the :math:`\LaTeX` table, the pd.DataFrame is also returned. By default,
        only the string with the :math:`\LaTeX` table is returned.

    Returns
    -------
    Union[str, Tuple[str, pd.DataFrame]]
        The :math:`\LaTeX` table, and optionally the pd.DataFrame associated.
    """

    available_mdps, available_agents = get_available_mdps_agents_prms_and_names(
        experiment_folder
    )

    table = pd.DataFrame(
        columns=pd.MultiIndex.from_tuples([("MDP", "")] + available_agents), dtype=str
    )
    agent_average_performance = {a: [] for a in available_agents}
    for i, (mdp_class_name, mdp_prm) in enumerate(available_mdps):
        row = [mdp_class_name]
        for k, (agent_class_name, agent_prm) in enumerate(available_agents):
            df, n_seeds = get_logs_data(
                experiment_folder,
                mdp_class_name,
                mdp_prm,
                agent_class_name,
                agent_prm,
            )
            values = df.loc[df.steps == df.steps.max(), indicator]
            if divide_by_total_number_of_time_steps:
                values /= df.steps.max() + 1
            row.append(f"${values.mean():.2f}\\pm{values.std():4.2f}$")
            agent_average_performance[agent_class_name, agent_prm].append(values.mean())
        if show_prm:
            row[0] = get_formatted_name(mdp_class_name, mdp_prm)

        scores = [float(re.findall(r"\$[0-9].[0-9]+", r)[0][1:]) for r in row[1:]]
        if "regret" in indicator or "steps_per_second" in indicator:
            best_scores = "$" + f"{(min(scores)):.2f}"
        elif "reward" in indicator:
            best_scores = "$" + f"{(max(scores)):.2f}"
        else:
            raise ValueError(f"I'm not sure whether min or max is best for {indicator}")
        for k in range(1, len(row)):
            row[k] = row[k].replace(best_scores, "$\\mathbf{" + best_scores[1:] + "}")

        row[0] = clear_agent_mdp_class_name(row[0])
        table.loc[len(table)] = row

    row = [r"\textit{Average}"]
    for c in table.columns[1:]:
        values = np.array(agent_average_performance[c])
        row.append(f"${values.mean():.2f}\\pm{values.std():4.2f}$")

    scores = [float(re.findall(r"\$[0-9].[0-9]+", r)[0][1:]) for r in row[1:]]
    if "regret" in indicator or "steps_per_second" in indicator:
        best_scores = "$" + f"{(min(scores)):.2f}"
    elif "reward" in indicator:
        best_scores = "$" + f"{(max(scores)):.2f}"
    else:
        raise ValueError(f"I'm not sure whether min or max is best for {indicator}")
    for k in range(1, len(row)):
        row[k] = row[k].replace(best_scores, "$\\mathbf{" + best_scores[1:] + "}")
    table.loc[len(table)] = row

    table.columns = pd.MultiIndex.from_tuples(
        [(clear_agent_mdp_class_name(n), p) for n, p in table.columns.values]
    )
    table = table.set_index("MDP")
    table_lat = table.copy()
    if show_prm:
        table_lat.index = [
            c.replace(c.split(" ")[0], " " * len(c.split(" ")[0]))
            if i > 0 and c.split(" ")[0] == table_lat.index[i - 1].split(" ")[0]
            else c
            for i, c in enumerate(table_lat.index)
        ]
    else:
        table_lat.index = [
            "" if i > 0 and c == table_lat.index[i - 1] else c
            for i, c in enumerate(table_lat.index)
        ]

    if not mdps_on_row:
        table = table.T
        table.columns = pd.MultiIndex.from_tuples(
            [(clear_agent_mdp_class_name(n), p) for n, p in available_mdps]
            + [(r"\textit{Average}", "")]
        )
        table_lat = table.copy()
        table_lat.index = [
            "" if i > 0 and c == table_lat.index[i - 1][0] else c
            for i, (c, p) in enumerate(table_lat.index.values)
        ]
    table_lat.index.name = None

    if print_table:
        with pd.option_context(
            "display.max_rows", None, "display.max_columns", None, "display.width", 500
        ):
            print(table)

    table_lat = table_lat.to_latex(escape=False).replace(
        r"\bottomrule", r"\arrayrulecolor{black!15}\midrule%"
    )
    if not show_prm:
        table_lat = table_lat.split("\n")
        table_lat.pop(3)
        table_lat = "\n".join(table_lat)

    # Add midrules between agents/mdps parameters
    table_lat = table_lat.split("\n")
    row_indices = (available_mdps if mdps_on_row else available_agents) + [
        (r"\textit{Average}", "")
    ]
    for i, (c, p) in reversed(list(enumerate(row_indices))):
        if i > 0 and c != row_indices[i - 1][0]:
            table_lat.insert(
                i + 4,
                r"\arrayrulecolor{black!"
                + f"{30 if 'Average' in row_indices[i][0] else 15}"
                + "}\midrule%",
            )
    table_lat = "\n".join(table_lat).replace("{l}", "{c}").replace("MiniGrid", "MG-")

    # Centering the columns with numbers
    columns_labels = re.findall(r"\{l+\}", table_lat)[0]
    table_lat = table_lat.replace(
        "l" * (len(columns_labels) - 2), "l" + "c" * (len(columns_labels) - 3)
    )

    if return_table:
        return table_lat, table
    return table


def get_latex_table_of_indicators(
    experiment_folder: str,
    indicators: List[str],
    show_prm_agent: bool = False,
    divide_by_total_number_of_time_steps: bool = True,
    print_table: bool = False,
    show_prm_mdp=True,
) -> str:
    r"""
    produces a latex table whose entries are the averages over the seeds of the indicator given in input for the
    results in the experiment folder.

    Parameters
    ----------
    experiment_folder : str
        The folder that contains the experiment logs, MDP configurations and agent configurations.
    indicators : List[str]
        The list of strings containing the performance indicators that will be shown in the plot. Check
        `MDPLoop.get_indicators()` to get a list of the available indicators.
    show_prm_agent : bool
        If True, the gin parameter config index is shown next to the agent class name. By default, it is not shown.
    divide_by_total_number_of_time_steps : bool
        If True, the value of the indicator is divided by the total number of time steps of agent/MDP interaction. By
        default, it is divided.
    print_table : bool
        If True, the table is printed.
    show_prm_mdp : bool
        If True, the gin parameter config index is shown next to the MDP class name. By default, it is shown.

    Returns
    -------
    str
        The :math:`\LaTeX` table.
    """

    assert all(
        ind in MDPLoop.get_indicators() for ind in indicators
    ), f"I received an invalid indicator, the available indicators are: {MDPLoop.get_indicators()}"

    available_mdps, available_agents = get_available_mdps_agents_prms_and_names(
        experiment_folder
    )
    # available_agents.insert(0, available_agents[0])

    table = pd.DataFrame(
        columns=[
            "MDP",
            "Agent",
            *map(format_indicator_name, indicators),
            r"\# completed seeds",
        ],
        dtype=str,
    )
    for i, (mdp_class_name, mdp_prm) in enumerate(available_mdps):
        for j, (agent_class_name, agent_prm) in enumerate(available_agents):
            row = [mdp_class_name, agent_class_name]

            df, n_seeds = get_logs_data(
                experiment_folder,
                mdp_class_name,
                mdp_prm,
                agent_class_name,
                agent_prm,
            )

            if "Continuous" in agent_class_name:
                df.normalized_cumulative_expected_reward = (
                    df.steps.max()
                    * (
                        df.cumulative_expected_reward
                        - df.worst_cumulative_expected_reward
                    )
                    / (
                        df.optimal_cumulative_expected_reward
                        - df.worst_cumulative_expected_reward
                    )
                )
                df.normalized_cumulative_reward = (
                    df.steps.max()
                    * (df.cumulative_reward - df.worst_cumulative_expected_reward)
                    / (
                        df.optimal_cumulative_expected_reward
                        - df.worst_cumulative_expected_reward
                    )
                )

            df[np.isclose(df, 0)] = 0

            values = df.loc[df.steps == df.steps.max(), indicators]
            if divide_by_total_number_of_time_steps:
                values /= df.steps.max() + 1
            row += [f"${v.mean():.2f}\\pm{v.std():.2f}$" for v in values.values.T]

            n_failed = get_n_failed_interactions(
                experiment_folder,
                mdp_class_name,
                mdp_prm,
                agent_class_name,
                agent_prm,
            )
            row.append(f"${n_seeds - n_failed}/{n_seeds}$")

            if show_prm_mdp:
                row[0] = get_formatted_name(mdp_class_name, mdp_prm).replace(
                    "MiniGrid", "MG-"
                )
            if show_prm_agent:
                row[1] = get_formatted_name(agent_class_name, agent_prm)
            row[0] = clear_agent_mdp_class_name(row[0])
            row[1] = clear_agent_mdp_class_name(row[1])
            table.loc[len(table)] = row

    table.MDP = [
        "" if i > 0 and c == table.MDP[i - 1] else c for i, c in enumerate(table.MDP)
    ]
    table.Agent = [
        "" if i > 0 and c == table.Agent[i - 1] else c
        for i, c in enumerate(table.Agent)
    ]
    table = table.set_index(["MDP", "Agent"])

    if print_table:
        with pd.option_context(
            "display.max_rows", None, "display.max_columns", None, "display.width", 500
        ):
            print(table)

    short_rule_indices = []
    long_rule_indices = []
    for i, (mdp_i, agent_i) in enumerate(table.index):
        if i > 0 and agent_i != "" and agent_i != table.index[i - 1][1]:
            if mdp_i == "":
                short_rule_indices.append(i + 5)
            else:
                long_rule_indices.append(i + 5)

    table_columns_len = len(table.columns)
    table = table.to_latex(escape=False).split("\n")
    for i in reversed(range(len(table))):
        if i in long_rule_indices:
            table.insert(
                i,
                r"\arrayrulecolor{black!15}\cmidrule{"
                + f"1-{1 + table_columns_len}"
                + "}",
            )
        elif i in short_rule_indices:
            table.insert(
                i,
                r"\arrayrulecolor{black!15}\cmidrule{"
                + f"2-{1 + table_columns_len}"
                + "}",
            )
    return "\n".join(table)
