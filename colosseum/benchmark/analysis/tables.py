from typing import List, Tuple, Union

import pandas as pd

from colosseum.benchmark.analysis.utils import (
    format_indicator_name,
    get_available_mdps_agents_prms_and_names,
    get_formatted_name,
    get_logs_data,
    get_n_failed_interactions,
)
from colosseum.utils.formatter import clear_agent_mdp_class_name


def get_latex_table_of_average_indicator(
    experiment_result_folder_path: str,
    indicator: str,
    show_prm: bool = False,
    divide_by_total_number_of_time_steps: bool = True,
    mdps_on_row: bool = True,
    print_table: bool = False,
    return_table : bool = False
) -> Union[str, Tuple[str, pd.DataFrame]]:
    """
    produces a latex table whose entries are the averages over the seeds of the indicator given in input for the
    results in the experiment folder.

    experiment_result_folder_path : str
        is the folder that contains the experiment logs, MDP configurations and agent configurations.
    indicator : str
        is a string representing the performance indicator that is reported in the table. Check
        MDPLoop.get_available_indicators() to get a list of the available indicators.
    show_prm : bool, optional
        checks whether to shoe the gin parameter config number next to the agent/MDP class name. By default, it is False.
    divide_by_total_number_of_time_steps : bool, optional
        checks whether to divide the value of the indicator by the total number of time steps of agent/MDP interaction.
        The default value is True.
    mdps_on_row : bool, optional
        checks whether to show MDPs or agents on the row indices. By default, it shows MDPs on row indices.
    print_table : bool, optional
        checks whether to additionally print the table.
    return_table : bool, optional
        checks whether to return the pd.DataFrame along with the latex string.
    """

    available_mdps, available_agents = get_available_mdps_agents_prms_and_names(
        experiment_result_folder_path
    )

    table = pd.DataFrame(
        columns=pd.MultiIndex.from_tuples([("MDP", "")] + available_agents), dtype=str
    )
    for i, (mdp_class_name, mdp_prm) in enumerate(available_mdps):
        row = [mdp_class_name]
        for agent_class_name, agent_prm in available_agents:
            df, n_seeds = get_logs_data(
                experiment_result_folder_path,
                mdp_class_name,
                mdp_prm,
                agent_class_name,
                agent_prm,
            )
            values = df.loc[df.steps == df.steps.max(), indicator]
            if divide_by_total_number_of_time_steps:
                values /= df.steps.max() + 1
            row.append(f"${values.mean():.2f}\\pm{values.std():.2f}$")
        if show_prm:
            row[0] = get_formatted_name(mdp_class_name, mdp_prm)
        row[0] = clear_agent_mdp_class_name(row[0])
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

    table_lat = table_lat.to_latex(escape=False)
    if not show_prm:
        table_lat = table_lat.split("\n")
        table_lat.pop(3)
        table_lat = "\n".join(table_lat)

    # Add midrules between agents/mdps parameters
    table_lat = table_lat.split("\n")
    row_indices = available_mdps if mdps_on_row else available_agents
    for i, (c, p) in reversed(list(enumerate(row_indices))):
        if i > 0 and c != row_indices[i - 1][0]:
            table_lat.insert(i + 4, r"\arrayrulecolor{black!15}\midrule%")
    table_lat = "\n".join(table_lat)

    if return_table:
        return table_lat, table
    return table.replace("{l}", "{c}")


def get_latex_table_of_indicators(
    experiment_result_folder_path: str,
    indicators: List[str],
    show_prm: bool = False,
    divide_by_total_number_of_time_steps: bool = True,
    print_table: bool = False,
) -> str:
    """
    produces a latex table whose entries are the averages over the seeds of the indicator given in input for the
    results in the experiment folder.

    experiment_result_folder_path : str
        is the folder that contains the experiment logs, MDP configurations and agent configurations.
    indicators : List[str]
        is a list of strings representing the performance indicators that is shown is reported in the table. Check
        MDPLoop.get_available_indicators() to get a list of the available indicators.
    show_prm : bool, optional
        checks whether to shoe the gin parameter config number next to the agent/MDP class name. By default, it is False.
    divide_by_total_number_of_time_steps : bool, optional
        checks whether to divide the value of the indicator by the total number of time steps of agent/MDP interaction.
        The default value is True.
    print_table : bool, optional
        checks whether to additionally print the table.
    """
    available_mdps, available_agents = get_available_mdps_agents_prms_and_names(
        experiment_result_folder_path
    )
    available_agents.insert(0, available_agents[0])

    table = pd.DataFrame(
        columns=[
            "MDP",
            "Agent",
            *map(format_indicator_name, indicators),
            r"\# completed in time",
        ],
        dtype=str,
    )
    for i, (mdp_class_name, mdp_prm) in enumerate(available_mdps):
        for j, (agent_class_name, agent_prm) in enumerate(available_agents):
            row = [mdp_class_name, agent_class_name]

            df, n_seeds = get_logs_data(
                experiment_result_folder_path,
                mdp_class_name,
                mdp_prm,
                agent_class_name,
                agent_prm,
            )
            values = df.loc[df.steps == df.steps.max(), indicators]
            if divide_by_total_number_of_time_steps:
                values /= df.steps.max() + 1
            row += [f"${v.mean():.2f}\\pm{v.std():.2f}$" for v in values.values.T]

            n_failed = get_n_failed_interactions(
                experiment_result_folder_path,
                mdp_class_name,
                mdp_prm,
                agent_class_name,
                agent_prm,
            )
            row.append(f"${n_seeds - n_failed}/{n_seeds}$")

            if show_prm:
                row[0] = get_formatted_name(mdp_class_name, mdp_prm)
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
