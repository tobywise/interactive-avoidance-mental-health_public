import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr, spearmanr, t
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


def print_demographics(df: pd.DataFrame) -> None:
    """
    Prints demographics information from a dataframe. It includes the number of unique subjects,
    the mean and standard deviation of the age, and the count of each gender category.

    Args:
        df (pd.DataFrame): A pandas dataframe containing at least the following columns:
                           'subjectID' with unique identifiers for subjects,
                           'age' with age values,
                           'gender' with gender categories encoded as 0, 1, 2, 3.

    Returns:
        None: This function prints the results and does not return anything.
    """
    # Print number of subjects
    print("Number of subjects = {}".format(len(df["subjectID"].unique())))

    # Print mean and standard deviation of age
    print("Mean (SD) age = {:.2f} ({:.2f})".format(df["age"].mean(), df["age"].std()))

    # Print gender counts
    gender_counts = (
        df["gender"]
        .value_counts()
        .rename({0: "Male", 1: "Female", 2: "Other", 3: "Prefer not to say"})
    )
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
