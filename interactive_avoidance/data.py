import json
import os
import re
from multiprocessing.sharedctypes import Value
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from maMDP.env_io import *
from maMDP.env_io import hex_environment_from_dict
from tqdm import tqdm


def load_all_data(
    ratings: bool = True,
    rt_prey: bool = True,
    rt_predator: bool = True,
    confidence: bool = True,
    prediction: bool = True,
    response: bool = True,
    questionnaire_data: pd.DataFrame = None,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Function to (optionally) load all data from the experiment, allowing for the user to specify which data types to load.

    Args:
        ratings (bool, optional): Whether to load rating data. Defaults to True.
        rt_prey (bool, optional): Whether to load prey reaction time data. Defaults to True.
        rt_predator (bool, optional): Whether to load predator reaction time data. Defaults to True.
        confidence (bool, optional): Whether to load confidence data. Defaults to True.
        prediction (bool, optional): Whether to load prediction data. Defaults to True.
        response (bool, optional): Whether to load response data. Defaults to True.
        questionnaire_data (pd.DataFrame, optional): Dataframe of questionnaire data, used to determine which subject IDs
                                                     to include. Only subjects present in the subjectID column of this dataframe
                                                     will be included in the loaded task data. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: Returns
        a tuple of dataframes, with the order being: rating_df, rt_prey_df, rt_predator_df, confidence_df,
        prediction_df, response_df. Data types not loaded will be None.
    """

    # Get subject IDs to include
    if questionnaire_data is not None:
        subjectIDs = questionnaire_data["subjectID"].unique()
    else:
        subjectIDs = None

    # Load data
    rating_df = pd.read_csv(f"data/task/rating_data.csv")
    rt_predator_df = pd.read_csv(f"data/task/rt_predator_data.csv")
    rt_prey_df = pd.read_csv(f"data/task/rt_prey_data.csv")
    confidence_df = pd.read_csv(f"data/task/confidence_data.csv")
    prediction_df = pd.read_csv(f"data/task/prediction_data.csv")
    response_df = pd.read_csv(f"data/task/response_data.csv")

    # Drop the 'exp' column from all dataframes
    dfs = [
        rating_df,
        rt_predator_df,
        rt_prey_df,
        confidence_df,
        prediction_df,
        response_df,
    ]
    for df in dfs:
        df.drop(columns=["exp"], inplace=True)

    # Filter subjectIDs
    if subjectIDs is not None:
        # Get initial number of subjects
        n_subs = len(rating_df["subjectID"].unique())
        # Remove subjects not in questionnaire data
        dfs = [
            df[df["subjectID"].isin(subjectIDs)] for df in dfs
        ]
        # Print number of subjects removed
        print(f"Removed {n_subs - len(dfs[0]['subjectID'].unique())} subjects not present in questionnaire data.")

    return dfs


def separate_predator_prey_data(
    response_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to separate predator and prey data from response data.

    Args:
        response_df (pd.DataFrame): Response data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Returns a tuple of dataframes, with the order being: prey_df, predator_df.
    """

    prey_df = response_df[response_df["agent"] == "prey"].reset_index(drop=True)
    predator_df = response_df[response_df["agent"] == "predator"].reset_index(drop=True)

    return prey_df, predator_df


def sort_dataframes(
    dataframes: List[pd.DataFrame],
) -> List[pd.DataFrame]:
    """
    Function to sort a list of dataframes by sujectID, condition, environment, trial, and response number.

    Args:
        dataframes (List[pd.DataFrame]): List of dataframes to sort.

    Returns:
        List[pd.DataFrame]: Returns a list of sorted dataframes.
    """

    sorted_dfs = []

    for df in dataframes:
        sorted_dfs.append(
            df.sort_values(
                ["subjectID", "condition", "env", "trial", "response_number"]
            ).reset_index(drop=True)
        )

    return sorted_dfs


def load_environment_data() -> pd.DataFrame:
    """
    Function to load environment data.

    Returns:
        pd.DataFrame: Dataframe of environment data.
    """

    envs = {}

    with open(
        "data/game_info/game_info.json",
        "r",
    ) as f:
        game_info = json.load(f)
    envs['cond1'] = [
        hex_environment_from_dict(env, ["Dirt", "Trees", "Reward"])
        for env in game_info["environments"]
    ]

    return envs


def load_factor_scores(n_solutions: int = 4, scale: bool = True) -> pd.DataFrame:
    """
    Function to load factor scores up to a given number of solutions.

    Args:
        n_solutions (int, optional): Number of solutions to load. Defaults to 4.
        scale (bool, optional): Whether to scale the factor scores. Defaults to True.

    Returns:
        pd.DataFrame: Dataframe of factor scores.
    """

    factor_score_dfs = []

    # Check that the factor scores have been saved
    if not os.path.exists("results/questionnaires/efa/hierarchical"):
        raise ValueError(
            "Factor scores not found. Please generate factor scores first."
        )

    for i in range(1, n_solutions):
        factor_score_df = pd.read_csv(
            "results/questionnaires/efa/hierarchical/factor_scores__{}.csv".format(i)
        )

        # Sort by subjectID
        factor_score_df = factor_score_df.sort_values("subjectID").reset_index(
            drop=True
        )

        # drop age and gender from all but the first
        if i > 1:
            factor_score_df = factor_score_df.drop(
                columns=["age", "gender", "subjectID"]
            )

        # Add prefix to ML columns
        factor_score_df = factor_score_df.rename(
            columns={
                c: "Sol{0}_{1}".format(i, c)
                for c in factor_score_df.columns
                if "ML" in c
            }
        )

        factor_score_dfs.append(factor_score_df)

    # merge factor score dfs
    factor_scores = pd.concat(factor_score_dfs, axis=1)

    # Scale factor scores
    if scale:
        for i in [i for i in factor_scores.columns if "Sol" in i] + ["age"]:
            # scale to mean 0, sd 1
            factor_scores[i] = (
                factor_scores[i] - factor_scores[i].mean()
            ) / factor_scores[i].std()

    # We only have a couple of people who indicated gender as anything other than male or female
    # which doesn't provide enough data to make meaningful inferences, and can cause problems
    initial_shape = factor_scores.shape[0]
    factor_scores = factor_scores[factor_scores["gender"].isin([0, 1])]
    print(
        "Dropped {} subjects with gender != 0 or 1.".format(
            initial_shape - factor_scores.shape[0]
        )
    )

    return factor_scores


def scale_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Utility function to scale data to have zero mean and unit variance, ignoring
    subjectID and gender columns.

    Args:
        df (pd.DataFrame): Dataframe to scale.

    Returns:
        pd.DataFrame: Scaled dataframe.
    """

    # Scale all variables from qdata
    for c in df.columns:
        if c not in ["subjectID", "gender"]:
            try:
                df[c] = (df[c] - df[c].mean()) / df[c].std()
            except:
                print("Could not scale {}".format(c))

    return df


def process_ratings(
    rating_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process the ratings DataFrame.

    Args:
        rating_df (pd.DataFrame): The original ratings DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - Cleaned and processed main ratings DataFrame.
            - Filtered DataFrame containing only confidence ratings.
            - DataFrame containing prey ratings, averaged within subject.
            - DataFrame containing mean confidence ratings, averaged within subject.
    """

    # Sort the original DataFrame by several columns and reset its index
    rating_df_cleaned = rating_df.sort_values(
        ["subjectID", "condition", "env"]
    ).reset_index(drop=True)

    # Filter out rows where the environment ('env') column is 0 (training environment)
    rating_df_cleaned = rating_df_cleaned[rating_df["env"] != 0]

    # Assign the 'condition' column to a new column named 'Reward weights'
    rating_df_cleaned["Reward weights"] = rating_df_cleaned["condition"]

    # Rename the 'rating' column to 'Rating' for consistency
    rating_df_cleaned = rating_df_cleaned.rename(columns={"rating": "Rating"})

    # Capitalize the 'feature' column entries for better presentation
    rating_df_cleaned["feature"] = rating_df_cleaned["feature"].str.capitalize()

    # Separate out entries where the feature is 'confidence'
    rating_confidence_df = rating_df_cleaned[
        rating_df_cleaned["feature"].str.lower() == "confidence"
    ]

    # Retain entries in rating_df_cleaned where the feature is not 'confidence'
    rating_df_cleaned = rating_df_cleaned[rating_df_cleaned["feature"] != "confidence"]

    # Shift the 'Rating' values by 50 to center around zero
    rating_df_cleaned["Rating"] = rating_df_cleaned["Rating"] - 50

    # Initialize an 'error' column with zeros
    rating_df_cleaned["error"] = 0

    # Compute the error for certain conditions and features related to trees, prey, and red
    for feature, expected_value in [("Trees", 50), ("Prey", 0), ("Red", 0)]:
        mask = (rating_df_cleaned["Reward weights"] == "cond1") & (
            rating_df_cleaned["feature"] == feature
        )
        rating_df_cleaned.loc[mask, "error"] = (
            expected_value - rating_df_cleaned.loc[mask, "Rating"]
        )

    # Rename conditions for clarity
    rating_df_cleaned["Reward weights"] = rating_df_cleaned["Reward weights"].replace(
        {"cond1": "Prefers trees"}
    )

    # Compute the absolute value of the error for each entry
    rating_df_cleaned["error_abs"] = np.abs(rating_df_cleaned["error"])

    # Extract and process prey ratings
    prey_ratings = (
        rating_df_cleaned[rating_df_cleaned["feature"] == "Prey"]
        .groupby(["subjectID"])
        .mean(numeric_only=True)
        .reset_index()
    )
    prey_ratings["prey_rating"] = prey_ratings["Rating"]
    prey_ratings["average_rating"] = (
        rating_df_cleaned.groupby(["subjectID"])
        .mean(numeric_only=True)
        .reset_index()["Rating"]
    )
    prey_ratings["prey_rating"] = (
        prey_ratings["prey_rating"] - prey_ratings["average_rating"]
    )
    prey_ratings["prey_rating"] = (
        prey_ratings["prey_rating"]
        - prey_ratings["prey_rating"].mean(numeric_only=True)
    ) / prey_ratings["prey_rating"].std()

    # Get mean confidence ratings
    rating_confidence_mean_df = (
        rating_confidence_df.groupby(["subjectID"])
        .mean(numeric_only=True)
        .reset_index()
    )
    rating_confidence_mean_df["confidence"] = rating_confidence_mean_df["Rating"]
    rating_confidence_mean_df["confidence_raw"] = rating_confidence_mean_df["Rating"]
    rating_confidence_mean_df["confidence"] = (
        rating_confidence_mean_df["confidence"]
        - rating_confidence_mean_df["confidence"].mean(numeric_only=True)
    ) / rating_confidence_mean_df["confidence"].std()

    # Get mean confidence variance
    rating_confidence_var_df = (
        rating_confidence_df.groupby(["subjectID"]).var(numeric_only=True).reset_index()
    )
    rating_confidence_var_df["confidence_var"] = rating_confidence_var_df["Rating"]
    rating_confidence_var_df["confidence_var"] = np.log(
        rating_confidence_var_df["confidence_var"] + 1e-5
    )  # log transform, add small amount to avoid inf
    rating_confidence_var_df["confidence_var"] = (
        rating_confidence_var_df["confidence_var"]
        - rating_confidence_var_df["confidence_var"].mean(numeric_only=True)
    ) / rating_confidence_var_df["confidence_var"].std()

    # Add confidence var column to mean df
    rating_confidence_mean_df["confidence_var"] = rating_confidence_var_df[
        "confidence_var"
    ]

    return (
        rating_df_cleaned,
        rating_confidence_df,
        prey_ratings,
        rating_confidence_mean_df,
    )
