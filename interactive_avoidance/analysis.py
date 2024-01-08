from typing import List
import pandas as pd
import pingouin as pg
import statsmodels.formula.api as smf
from stats_utils.regression.analysis import add_bootstrap_methods_to_ols


def run_exploratory_models(
    parameter_names: List[str],
    data: pd.DataFrame,
    measures: List[str] = ["AQ_10", "STICSA_T", "LSAS", "PHQ_8", "GAD_7"],
    n_bootstraps: int = 2000,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Run exploratory models for each parameter and measure of interest

    Args:
        parameter_names (List[str]): List of parameter names
        data (pd.DataFrame): Dataframe containing parameter and measure data
        measures (List[str], optional): List of measures of interest. Defaults to ["AQ_10", "STICSA_T", "LSAS", "PHQ_8", "GAD_7"].
        n_bootstraps (int, optional): Number of bootstraps to run. Defaults to 2000.
        alpha (float, optional): Alpha value for confidence intervals. Defaults to 0.05.

    Returns:
        pd.DataFrame: Dataframe containing model results

    """

    # Initialize output dataframe
    output_df = {
        "parameter": [],
        "measure": [],
        "coef": [],
        "coef_se": [],
        "t": [],
        "p": [],
        "ci_lower": [],
        "ci_upper": [],
    }

    # Loop over parameters
    for p in parameter_names:
        # Loop over measures of interest
        for v in measures:
            # Specify model
            if p == "confidence":
                model = smf.ols(
                    "{0} ~ age + gender + motivation + error_abs + {1}".format(p, v),
                    data=data,
                )
            else:
                model = smf.ols(
                    "{0} ~ age + gender + motivation + {1}".format(p, v), data=data
                )

            # Fit model
            fitted_model = model.fit()

            # Replace the class with the bootstrap results class
            fitted_model = add_bootstrap_methods_to_ols(fitted_model)

            # Run bootstrap
            fitted_model.bootstrap(n_bootstraps)
            _ = fitted_model.conf_int_bootstrap(alpha=alpha)

            # Add to output dataframe
            output_df["parameter"].append(p)
            output_df["measure"].append(v)
            output_df['coef'].append(fitted_model.params[v])
            output_df['coef_se'].append(fitted_model.bse[v])
            output_df["t"].append(fitted_model.tvalues[v])
            output_df["p"].append(fitted_model.pvalues_bootstrap[v])
            output_df["ci_lower"].append(fitted_model.conf_int_bootstrap().loc[v, 0])
            output_df["ci_upper"].append(fitted_model.conf_int_bootstrap().loc[v, 1])

    output_df = pd.DataFrame(output_df)

    # FDR correct p values
    output_df["p_fdr"] = pg.multicomp(output_df["p"].values, method="fdr_bh")[1]

    return output_df
