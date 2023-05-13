#
# This script process the resulting json files and generates
# latex tables.
#
from argparse import ArgumentParser
import sys

import yaml

sys.path.append('scripts-models')

import json
from os import listdir
from os.path import isfile, join
import os

import pandas as pd
# import modelset.dataset as ds
# from common import read_dataset

import numpy as np
import seaborn as sns


def read_report_config(report_config):
    with open(report_config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            results = config["results"]
            if results is None:
                raise Exception("No results in config")

            config["results"] = os.path.join(os.path.dirname(report_config), results)
            return config
        except yaml.YAMLError as exc:
            print(exc)
            exit()


def main_with_args(args):
    tasks = args.tasks.split(",")
    report_config = read_report_config(args.config_file)
    results = read_files(report_config["results"], tasks)

    rows = []
    for data in results:
        conf = data["configuration"]
        metric_score_pairs = {}

        if isinstance(conf["task"]["metric"], str):
            conf["task"]["metric"] = [conf["task"]["metric"]]

        for i, metric in enumerate(conf["task"]["metric"]):
            if "score_test" in data["results"]:
                value = data["results"]["score_test"]
                score = value[i] if isinstance(value, list) else value
            elif "mean_all_scores" in data["results"]:
                value = data["results"]["mean_all_scores"]
                # TODO: Change this when computing the clustering value, because it introduces a very weird thing
                if isinstance(value, dict):
                    value = list(value.values())[0]

                score = value[i] if isinstance(value, list) else value
            else:
                # Assume it is a k-fold
                score = data["results"]["results_best_hyperparameter"]["score_folds_mean"][i]

            metric_score_pairs[metric] = score

        result = {
            "model": conf["combination"]["ml_model"],
            "features": conf["combination"]["encoding_features"]
        }
        result.update(metric_score_pairs)
        rows.append(result)

    df = pd.DataFrame(rows)

    if "sort_by" in report_config:
        df.sort_values(by=report_config["sort_by"], inplace=True, ascending=False)

    # df.set_index(["model", "features"], inplace=True)
    print(df)

    # Convert the pandas dataframe df to a LaTeX table
    # and save it to a file
    if args.output:
        replacements = report_config.get("header", {})
        caption = report_config.get("caption", None)
        label = report_config.get("label", None)

        with open(os.path.join(args.output, "result.tex"), 'w') as ofile:
            # Transform dataframe to latex table
            # When transforming to latex replace the column names with the replacements
            # and round the accuracy to 3 decimal places
            latex_table = df.to_latex(
                index=False,
                float_format="{:0.3f}".format,
                column_format="l" + "c" * len(df.columns),
                header=[replacements.get(c, c) for c in df.columns],
                caption=caption,
                label=label
            )
            ofile.write(latex_table)

    #
    # files_ecore = [join(args.result, f) for f in listdir(args.result) if isfile(join(args.result, f)) and "ecore" in f]
    #
    # print("\n\nAll ECORE\n")
    # print_dataset_info(files_ecore, "all", join(args.output, "ecore-dups.tex"))
    # print("\n\nNo duplicates ECORE\n")
    # print_dataset_info(files_ecore, "noDups", join(args.output, "ecore-no-dups.tex"))
    #
    # files_uml = [join(args.result, f) for f in listdir(args.result) if isfile(join(args.result, f)) and "uml" in f]
    #
    # print("\n\nAll UML\n")
    # print_dataset_info(files_uml, "all", join(args.output, "uml-dups.tex"))
    # print("\n\nNo duplicates UML\n")
    # print_dataset_info(files_uml, "noDups", join(args.output, "uml-no-dups.tex"))


def read_files(results_folder, tasks):
    files = [join(results_folder, f) for f in listdir(results_folder) if
             isfile(join(results_folder, f)) and f.endswith(".json")]
    jsons = []
    for f in files:
        with open(f, "r") as f:
            data = json.load(f)
            task = data["configuration"]["task"]["task_name"]
            if task in tasks:
                jsons.append(data)
    return jsons


def main():
    parser = ArgumentParser(description='Analyse the results')
    parser.add_argument("--tasks", dest="tasks", help="comma-separated list of tasks", required=True)
    parser.add_argument("--config", dest="config_file", help="configuration file for the report", required=True)
    parser.add_argument("--output-folder", dest="output", help="folder to place the generated files", required=True)
    args = parser.parse_args()

    main_with_args(args)


if __name__ == "__main__":
    main()
