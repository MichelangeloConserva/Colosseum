import os
import uuid
from glob import glob
from typing import TYPE_CHECKING, Union

import gin
import yaml
from tqdm import trange

from colosseum.utils.miscellanea import ensure_folder

if TYPE_CHECKING:
    from colosseum.mdps import MDP


def find_hardness_report_file(
    mdp: "MDP", hardness_reports_folder="hardness_reports"
) -> Union[str, None]:
    """
    tries to find a previously calculated hardness report for an MDP instance.

    Parameters
    ----------
    mdp : MDP
        the MDP instance for which it will be looking for the hardness report.
    hardness_reports_folder : str
        the folder in which the hardness reports are stored

    Returns
    -------
    the report if found and None otherwise.
    """
    from colosseum.utils.logging import cleaner

    seed_report = glob(f"{ensure_folder(hardness_reports_folder)}{type(mdp).__name__}_*.yml")
    for existing_report in seed_report:
        if type(mdp).__name__ not in existing_report:
            continue
        with open(existing_report, "r") as f:
            report = yaml.load(f, yaml.Loader)
        same_mdp = True
        for k, v in report["MDP parameters"].items():
            if not same_mdp:
                break
            same_mdp = cleaner(getattr(mdp, k)) == v
        for k, v in report["MDP graph metrics"].items():
            if not same_mdp:
                break
            same_mdp = mdp.graph_metrics[k] == v
        if same_mdp:
            return existing_report
    return None


def store_hardness_report(mdp: "MDP") -> str:
    """
    stores the hardness report for an MDP instance and return the path of the file.
    """
    from colosseum.utils.logging import clean_for_storing

    identifier = str(uuid.uuid1())
    while (
        len(glob(f"hardness_reports{os.sep}{type(mdp).__name__}_{identifier}.yml")) != 0
    ):
        identifier = str(uuid.uuid1())
    f_path = f"hardness_reports{os.sep}{type(mdp).__name__}_{identifier}.yml"

    measures_of_hardness = mdp.measures_of_hardness
    parameters = mdp.parameters
    graph_metrics = mdp.graph_metrics
    with open(f_path, "w") as outfile:
        yaml.dump(
            {
                "MDP name": type(mdp).__name__,
                "MDP parameters": clean_for_storing(parameters),
                "MDP graph metrics": clean_for_storing(graph_metrics),
                "MDP measure of hardness": clean_for_storing(measures_of_hardness),
            },
            outfile,
            default_flow_style=False,
            sort_keys=False,
        )
    return f_path


def create_missing_hardness_reports(
    experiments_mdp_configurations,
    num_seeds: int,
    verbose: Union[int, bool] = False,
    hardness_reports_folder="hardness_reports",
):
    """
    given an experiments MDP configurations, it checks if any of the MDPs is missing a hardness reports and stores it in
    the given folder.
    """

    from colosseum.experiments.utils import apply_gin_config

    # Removing clones
    reports = dict()
    for existing_report in glob("hardness_reports/*"):
        with open(existing_report, "r") as f:
            reports[existing_report] = "".join(f.readlines())
    for r, v in reports.items():
        for rr, vv in reports.items():
            if r == rr:
                continue
            if v == vv:
                if os.path.isfile(rr):
                    os.remove(rr)

    # Removing incomplete ones
    for existing_report in glob(f"{ensure_folder(hardness_reports_folder)}*"):
        with open(existing_report, "r") as f:
            report = yaml.load(f, yaml.Loader)
        if (
            report is None
            or "MDP parameters" not in report
            or "MDP graph metrics" not in report
            or "MDP measure of hardness" not in report
        ):
            os.remove(existing_report)

    n = 0
    for (
        result_folder,
        experiments_mdp_classes_and_gin_scopes,
        gin_config_files_paths,
    ) in experiments_mdp_configurations:

        for mdp_class, mdp_scopes in experiments_mdp_classes_and_gin_scopes.items():
            for mdp_scope in mdp_scopes:
                for seed in range(num_seeds):
                    n += 1

    loop = (
        trange(n, desc="Creating missing hardness reports", mininterval=5)
        if verbose
        else None
    )
    for (
        result_folder,
        experiments_mdp_classes_and_gin_scopes,
        gin_config_files_paths,
    ) in experiments_mdp_configurations:
        apply_gin_config(gin_config_files_paths)
        for mdp_class, mdp_scopes in experiments_mdp_classes_and_gin_scopes.items():
            n = 0
            for mdp_scope in mdp_scopes:
                with gin.config_scope(mdp_scope):
                    for seed in range(num_seeds):
                        if verbose:
                            loop.set_description(
                                f"{mdp_class.__name__}: {n / len(mdp_scopes) / num_seeds * 100:.2f}"
                            )
                        mdp = mdp_class(
                            seed=seed,
                            verbose=verbose
                            if type(verbose) == bool
                            else bool(verbose - 1),
                        )
                        if find_hardness_report_file(mdp) is None:
                            store_hardness_report(mdp)

                        if verbose:
                            loop.update(1)
                            loop.refresh()


def mean_metrics_report(reports_files):
    mean_reports = None
    for report in reports_files:
        with open(report, "r") as r:
            if mean_reports is None:
                mean_reports = yaml.load(r, yaml.Loader)
            else:
                for k, v in yaml.load(r, yaml.Loader).items():
                    if type(v) == dict:
                        for kk, vv in v.items():
                            if type(vv) in [int, float]:
                                mean_reports[k][kk] += vv
    for k, v in mean_reports.items():
        if type(v) == dict:
            for kk, vv in v.items():
                if type(vv) in [int, float]:
                    mean_reports[k][kk] += mean_reports[k][kk] / len(reports_files)
    return mean_reports, yaml.dump(mean_reports, sort_keys=False).replace(
        "  ", "\t"
    ).replace("\nMDP", "\n\n  MDP").replace("MDP name: ", "")
