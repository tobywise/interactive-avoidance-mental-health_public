import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr, spearmanr, t
from stats_utils.regression.analysis import ModelOutput
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.anova import anova_lm
from tqdm import tqdm


def check_directories():
    """
    Checks if the script is being run in the root directory and if the required data is present.
    Raises RuntimeError if any check fails.
    """
    # Check if the 'notebooks' directory exists
    if not os.path.isdir("notebooks"):
        # If we're currently in "notebooks", move one directory up
        if os.path.isdir("../notebooks"):
            print("Changing directory to root directory of repository...")
            os.chdir("..")
        else:
            raise RuntimeError(
                "You must run this notebook from the root directory of the repository, otherwise paths will break. You are currently in {}".format(
                    os.getcwd()
                )
            )

    # Check if the 'data' directory exists and is not empty
    if not os.path.isdir("data") or len(os.listdir("data")) == 0:
        raise RuntimeError(
            "You must download the data files from OSF and place them in the /data directory before running this notebook."
        )

    # Check if the 'figures' directory exists and create it if not
    if not os.path.isdir("figures"):
        os.mkdir("figures")


def print_demographics(df: pd.DataFrame, gender_mapping: Dict = None) -> None:
    """
    Prints demographics information from a dataframe. It includes the number of unique subjects,
    the mean and standard deviation of the age, and the count of each gender category.

    Args:
        df (pd.DataFrame): A pandas dataframe containing at least the following columns:
                           'subjectID' with unique identifiers for subjects,
                           'age' with age values,
                           'gender' with gender categories encoded as 0, 1, 2, 3.
        gender_mapping (Dict, optional): A dictionary mapping numeric values to string
                                         gender categories.

    Returns:
        None: This function prints the results and does not return anything.
    """
    # Print number of subjects
    print("Number of subjects = {}".format(len(df["subjectID"].unique())))

    # Print mean and standard deviation of age
    print("Mean (SD) age = {:.2f} ({:.2f})".format(df["age"].mean(), df["age"].std()))

    # Set default mapping if none is provided
    if gender_mapping is None:
        gender_mapping = {0: "Male", 1: "Female", 2: "Other", 3: "Prefer not to say"}

    # Print gender counts
    gender_counts = df["gender"].value_counts().rename(gender_mapping)
    counts_str = ", ".join(
        [f"{count} {gender}" for gender, count in gender_counts.items()]
    )
    print(f"Gender counts: {counts_str}")


def plot_regression(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_var: str,
    y_var: str,
    xlabel: str,
    ylabel: str = "",
) -> None:
    """
    Plot a regression plot on the specified axis.

    Args:
        ax (plt.Axes): Axis object on which the regression plot will be drawn.
        data (pd.DataFrame): Dataframe containing the data to plot.
        x_var (str): Column name in the dataframe to be used for the x-axis.
        y_var (str): Column name in the dataframe to be used for the y-axis.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis. Defaults to an empty string.

    Returns:
        None: The function modifies the `ax` object in-place.
    """

    sns.regplot(x=x_var, y=y_var, data=data, scatter_kws={"alpha": 0.2, "s": 3}, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def calculate_and_print_correlations(
    data: pd.DataFrame, x_var: str, y_var: str, label: str
) -> None:
    """
    Calculate and print both Pearson and Spearman correlations for given data columns.

    Args:
        data (pd.DataFrame): Dataframe containing the data for correlation calculation.
        x_var (str): Column name in the dataframe to be used as x data.
        y_var (str): Column name in the dataframe to be used as y data.
        label (str): Descriptive label for the correlation.

    Returns:
        None: The function prints correlations to the console.
    """

    # Calculate Pearson correlation
    corr_pearson, p_pearson = pearsonr(data[x_var], data[y_var])

    # Calculate Spearman correlation
    corr_spearman, p_spearman = spearmanr(data[x_var], data[y_var])

    # Print results
    print(f"{label} (Pearson): r = {corr_pearson:.2f}, p = {p_pearson:.3f}")
    print(f"{label} (Spearman): rho = {corr_spearman:.2f}, p = {p_spearman:.3f}")


def coefficients_to_dataframe(
    model_output: ModelOutput, model_index: int = -1, rename_dict: dict = None
) -> pd.DataFrame:
    """
    Converts a statsmodels OLS model coefficients table from a ModelOutput summary into a pandas DataFrame.

    Args:
        model_output (ModelOutput): The ModelOutput object whose summary is to be converted.
        model_index (int, optional): Index of the model in the summary to convert. Defaults to -1.
        rename_dict (dict, optional): Dictionary mapping original variable names to new names. Defaults to None.

    Returns:
        pd.DataFrame: A pandas DataFrame representation of the specified summary table.
    """
    # Get the model summary
    summary = model_output.models[model_index].summary()

    # Extract the specified table (SimpleTable object) from the summary
    table = summary.tables[1]

    # Extracting the data from the SimpleTable and converting it to a DataFrame
    data = table.data
    headers = data[0]
    rows = data[1:]

    # Converto to DataFrame
    df = pd.DataFrame(rows, columns=headers)

    # Rename row names in index based on rename_dict
    if rename_dict is not None:
        df = df.replace(rename_dict)

    return df


def dataframe_to_markdown(df: pd.DataFrame, round_dict: dict, rename_dict: dict) -> str:
    """
    Processes a pandas DataFrame by rounding specified columns, renaming columns with LaTeX formatted names,
    and converting the DataFrame to a markdown table string.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        round_dict (dict): A dictionary specifying the number of decimal places for each column to round to.
                           Example: {"column1": 2, "column2": 3}
        rename_dict (dict): A dictionary specifying the new column names with LaTeX formatting.
                            Example: {"column1": "$column_{1}$", "column2": "$column_{2}$"}

    Returns:
        str: A string representing the DataFrame in markdown format.

    Example:
        df = pd.DataFrame(...)
        round_dict = {"df_resid": 0, "ssr": 2, "ss_diff": 2, "F": 2, "Pr(>F)": 3}
        rename_dict = {"df_resid": "$df_{R}$", "ssr": "$SS_{R}$", "ss_diff": "$SS_{diff}$", "F": "$F$", "Pr(>F)": "$p$"}
        latex_str = dataframe_to_latex(df, round_dict, rename_dict)
    """

    # Create a copy of the DataFrame
    df = df.copy()

    # Get rounding precision for each column as a tuple in the column order, as a formatting string
    precisions = tuple([f".{round_dict.get(col, 0)}f" for col in df.columns])

    # Rename the columns
    df_renamed = df.rename(columns=rename_dict)

    # Convert to Markdown string
    return df_renamed.to_markdown(index=False, floatfmt=precisions)


def get_last_table_number(filename: str) -> int:
    """
    Reads a markdown file and returns the last table number found.

    Args:
        filename (str): The name of the markdown file.

    Returns:
        int: The last table number or 0 if no tables are found.
    """
    if not os.path.exists(filename):
        return 0

    last_table_number = 0
    with open(filename, "r") as file:
        for line in file:
            if line.startswith("*Table S"):
                try:
                    # Extract table number
                    table_number = int(line.split(".")[0].split("S")[-1].strip())
                    last_table_number = max(last_table_number, table_number)
                except ValueError:
                    continue
    return last_table_number


def dataframes_to_markdown(
    dfs: List[pd.DataFrame],
    captions: List[str],
    round_dicts: List[dict],
    rename_dicts: List[dict],
    filename: str,
    prepend_string: str = "",
    append: bool = False,
) -> None:
    """
    Combines multiple pandas DataFrames into a single Markdown string with separate tables and captions,
    using the dataframe_to_markdown function for each DataFrame, and exports or appends the result
    to a Markdown (.md) file.

    Args:
        dfs (list[pd.DataFrame]): List of pandas DataFrames to be converted.
        captions (list[str]): List of captions for each table.
        round_dicts (list[dict]): List of dictionaries for rounding columns for each DataFrame.
        rename_dicts (list[dict]): List of dictionaries for renaming columns for each DataFrame.
        filename (str): Name of the output Markdown file (should end in .md).
        prepend_string (str): String to prepend to the Markdown string.
        append (bool): If True, append to an existing file. If False, overwrite the file.

    Returns:
        None: This function writes to a file and does not return anything.
    """

    # Create an empty string to store the Markdown
    markdown_string = ""

    # Prepend the string if provided
    if prepend_string != "":
        markdown_string += prepend_string + "\n\n"

    # Check if lists are of equal length
    if not all(len(lst) == len(dfs) for lst in [captions, round_dicts, rename_dicts]):
        raise ValueError("All lists must be of the same length as dfs")

    # Get the last table number if appending
    start_table_number = get_last_table_number(filename) if append else 0

    # Loop through the dataframes
    for i, df in enumerate(dfs):
        table_number = start_table_number + i + 1
        print(dataframe_to_markdown(df, round_dicts[i], rename_dicts[i]))
        markdown_string += dataframe_to_markdown(df, round_dicts[i], rename_dicts[i])
        markdown_string += f"\n\n*Table S{table_number}. {captions[i]}*\n\n"

    # Determine the mode for opening the file
    file_mode = "a" if append and os.path.exists(filename) else "w"

    # Inform the user about the file operation
    if file_mode == "a":
        print(f"Appending to existing file: {filename}")
    else:
        print(f"Creating or overwriting file: {filename}")

    # Write or append to the markdown file
    with open(filename, file_mode) as file:
        file.write(markdown_string)
