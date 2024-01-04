"""
Runs model fitting for subjects' predictions about the predator's actions.
"""

import contextlib
from copy import copy, deepcopy
from typing import Dict, List, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from maMDP.algorithms.action_selection import ActionSelector, MaxActionSelector
from maMDP.algorithms.dynamic_programming import ValueIteration
from maMDP.algorithms.policy_learning import *
from maMDP.algorithms.policy_learning import (
    BaseGeneralPolicyLearner,
    TDGeneralPolicyLearner,
)
from maMDP.mdp import MDP, HexGridMDP
from numpy.lib.function_base import diff
from numpy.lib.type_check import nan_to_num
from scipy.optimize import differential_evolution, minimize
from scipy.stats import zscore
from tqdm import tqdm


def minmax_scale(X, min_val=0, max_val=1):
    """Adapted copy of https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html"""

    X[np.isinf(X)] = np.nan
    X_std = (X - np.nanmin(X, axis=1)[:, None]) / (
        (np.nanmax(X, axis=1)[:, None] - np.nanmin(X, axis=1)[:, None]) + 1e-10
    )
    X_scaled = X_std * (max_val - min_val) + min_val

    return X_scaled


def check_mdp_equal(mdp1, mdp2):
    """Checks whether two MDPs are the same based on features and transition matrix"""

    if not isinstance(mdp1, MDP) or not isinstance(mdp2, MDP):
        return False

    elif (mdp1.features == mdp2.features).all() and (mdp1.sas == mdp2.sas).all():
        return True
    else:
        return False


class VIPolicyLearner(BaseGeneralPolicyLearner):
    """Used to make VI-based action prediction work more nicely"""

    def __init__(
        self,
        VI_instance: ValueIteration,
        reward_weights: np.ndarray,
        refit_on_new_mdp: bool = True,
        caching: bool = False,
    ):
        """
        Estimates Q values for actions

        Args:
            VI_instance (ValueIteration): Instantiated instance of the value iteration algorithm.
            reward_weights (np.ndarray): Reward weights used to calculate reward function for VI.
            refit_on_new_mdp (bool, optional): If true, refits the model whenever fit() is provided with an MDP that differs
            from the previous one.
        """

        self.VI = VI_instance
        self.reward_weights = reward_weights
        self.q_values = None
        self.refit_on_new_mdp = refit_on_new_mdp
        self.caching = caching
        self.previous_mdp = None
        self.previous_mdp_q_values = {}

    def reset(self):
        self.q_values = None

    def fit(self, mdp: MDP, trajectories: list):
        """Estimates Q values

        Args:
            mdp (MDP): MDP in which the agent is acting.
            trajectories (list): List of trajectories. Not used but retained for compatibility.
        """

        if not check_mdp_equal(mdp, self.previous_mdp):
            cached_found = False
            # TODO seems like there are things in cache without having added to cache
            if self.caching:
                for m, q in self.previous_mdp_q_values.items():
                    if check_mdp_equal(mdp, m):
                        self.q_values = q
                        cached_found = True
            if not cached_found:
                self.VI.fit(mdp, self.reward_weights, None, None)
                self.q_values = self.VI.q_values
                if self.caching:
                    self.previous_mdp_q_values[mdp] = self.q_values.copy()

            self.previous_mdp = mdp

    def get_q_values(self, state: int) -> np.ndarray:
        """
        Returns Q values for each action in a given state.

        Args:
            state (int): State to get Q values for

        Returns:
            np.ndarray: Q values for each action in the provided state.
        """

        Q_values = self.q_values[state]
        Q_values[np.isinf(Q_values)] = 0  # Remove inf for invalid actions

        return Q_values

    def copy(self):
        # Copy without previous MDP to avoid picking error
        new_model = VIPolicyLearner(
            deepcopy(self.VI), copy(self.reward_weights), self.refit_on_new_mdp
        )
        new_model.q_values = self.q_values.copy()
        new_model.previous_mdp = None

        return new_model


class CombinedPolicyLearner(BaseGeneralPolicyLearner):
    def __init__(
        self,
        model1: BaseGeneralPolicyLearner,
        model2: BaseGeneralPolicyLearner,
        W: float = 0.5,
        scale: bool = True,
    ):
        """
        Produces a weighted combination of Q value estimates from two models.

        Args:
            model1 (BaseGeneralPolicyLearner): First model.
            model2 (BaseGeneralPolicyLearner): Second model.
            W (float, optional): Weighting parameter, lower values give Model 1 more weight. Defaults to 0.5.
            scale (bool, optional): If true, Q values from each model are minmax scaled to enable comparability between the two models.
            Defaults to True.
        """

        self.model1 = model1
        self.model2 = model2

        if not 0 <= W <= 1:
            raise ValueError("W must be between 0 and 1 (inclusive)")

        self.W = W
        self.scale = scale

    def reset(self):
        self.model1.reset()
        self.model2.reset()

    def fit(self, mdp: Union[MDP, List[MDP]], trajectories: list):
        """Estimates Q values

        Args:
            mdp (Union[MDP, List[MDP]]): MDP in which the agent is acting.
            trajectories (list): List of trajectories. NOTE: is only implemented for a single pair of states.

        """

        if len(trajectories) > 1:
            raise NotImplementedError()
        if len(trajectories[0]) > 2:
            raise NotImplementedError()

        # Fit both models
        self.model1.fit(mdp, trajectories)
        self.model2.fit(mdp, trajectories)

        self.fit_complete = True

    def get_q_values(self, state: int) -> np.ndarray:
        """
        Returns Q values for each action in a given state.

        Args:
            state (int): State to get Q values for

        Returns:
            np.ndarray: Q values for each action in the provided state.
        """

        model1_Q = self.model1.get_q_values(state)
        model2_Q = self.model2.get_q_values(state)

        if self.scale:
            model1_Q_scaled = minmax_scale(model1_Q[None, :]).squeeze()
            model2_Q_scaled = minmax_scale(model2_Q[None, :]).squeeze()
        else:
            model1_Q_scaled = model1_Q
            model2_Q_scaled = model2_Q

        overall_Q = (1 - self.W) * np.nan_to_num(
            model1_Q_scaled
        ) + self.W * np.nan_to_num(model2_Q_scaled)

        return overall_Q

    def copy(self):
        model1_copy = self.model1.copy()
        model2_copy = self.model2.copy()

        new_model = CombinedPolicyLearner(model1_copy, model2_copy, self.W, self.scale)
        return new_model


def nan_softmax(x: np.ndarray, return_nans: bool = False) -> np.ndarray:
    """
    Softmax function, ignoring NaN values. Expects a 2D array of Q values at different observations.

    This is important because for some states certain actions are invalid, so we need to ignore them when
    calculating the softmax rather than letting them influence the probabilities of other actions.

    Args:
        x (np.ndarray): Array of action values. Shape = (observations, actions)
        return_nans (bool, optional): If true, NaN values are returned as NaN,
        otherwise they are replaced with zeros. Defaults to False.

    Returns:
        np.ndarray: Array of probabilities, ignoring NaN values.
    """

    if not x.ndim == 2:
        raise AttributeError("Only works on 2D arrays")

    x_ = np.exp(x) / np.nansum(np.exp(x), axis=1)[:, None]

    if return_nans:
        return x_
    else:
        x_[np.isnan(x_)] = 0
        return x_


def prediction_likelihood(q: np.ndarray, pred_actions: List[int]) -> float:
    """
    Calculates categorical likelihood.

    Args:
        q (np.ndarray): Array of Q values for each action at each observation, shape (observations, actions).
        pred_actions (List[int]): List of observed actions, one per observation.

    Returns:
        float: Log likelihood of the observed actions given the provided Q values.
    """

    assert (
        len(pred_actions) == q.shape[0]
    ), "Different numbers of predicted actions ({0}) and Q values ({1})".format(
        len(pred_actions), q.shape[0]
    )

    # Convert predicted actions to int
    pred_actions = np.array(pred_actions).astype(int).tolist()

    # Scale Q values so they're all on the same scale regardless of the model
    q = minmax_scale(
        q, max_val=5
    )  # Using 5 (arbitrarily) has same effect as reducing decision noise

    action_p = nan_softmax(q, return_nans=True)

    logp = np.nansum(np.log((action_p[range(len(pred_actions)), pred_actions]) + 1e-8))
    if np.isinf(logp):
        raise ValueError("Inf in logp")

    return logp


def fit_policy_learning(X: Tuple[float], *args: List) -> float:
    """
    Fits a policy learning model. Intended for use with scipy optimization functions.

    Args:
        X (Tuple): Learning rate
        args (List): Other arguments. 1: Predator trajectories, 2: MDPs,
        3: Subject's predicted actions, 4: Whether to use generalisation kernel, 5: Whether to reset the model
        for each environment

    Returns:
        float: Log likelihood
    """

    alpha = X

    if np.isnan(alpha):
        alpha = 0.001

    predator_t, target_mdp, predicted_a, kernel, env_reset = args

    _, Q_estimates, _, _ = action_prediction_envs(
        predator_t,
        target_mdp,
        TDGeneralPolicyLearner(learning_rate=alpha, kernel=kernel, decay=0),
        action_selector=MaxActionSelector(seed=123),
        env_reset=env_reset,
    )

    logp = prediction_likelihood(np.vstack(Q_estimates), np.hstack(predicted_a))

    return -logp


def simulate_policy_learning(X: Tuple[float], args: List) -> Union[List, List]:
    """
    Simulates choices made by a policy learning model. Takes arguments in the same form as the corresponding fitting function.

    Args:
        X (Tuple): Learning rate.
        args (List): Other arguments. 1: Predator trajectories, 2: MDPs,
        3: Subject's predicted actions, 4: Whether to use generalisation kernel, 5: Whether to reset the model
        for each environment

    Returns:
        Union[List, List]: List of predicted states, list of predicted actions
    """

    alpha = X

    predator_t, target_mdp, predicted_a, kernel, env_reset = args

    _, _, simulated_predictions, simulated_predicted_actions = action_prediction_envs(
        predator_t,
        target_mdp,
        TDGeneralPolicyLearner(learning_rate=alpha, kernel=kernel, decay=0),
        action_selector=MaxActionSelector(seed=123),
        env_reset=env_reset,
    )

    return simulated_predictions, simulated_predicted_actions


def fit_combined_model(W: float, *args: List) -> float:
    """
    Fits a combined policy learning/value iteration model without estimating a learning rate for the policy learner.
    Intended for use with scipy optimization functions.

    Args:
        W (float): Weighting parameter, higher = higher weighting of policy learning.
        args (List): Other arguments. 1: Predator trajectory, 2: MDP,
        3: Subject's predicted actions, 4: Value iteration model, 5: Learning rate, 6: Whether
        to reset the model for each environment

    Returns:
        float: Log likelihood
    """

    # Avoid NaNs causing problems
    if np.isnan(W):
        W = 0.001

    predator_t, target_mdp, predicted_a, model1, learning_rate, env_reset = args

    model2 = TDGeneralPolicyLearner(learning_rate=learning_rate, decay=0)

    _, Q_estimates, _, _ = action_prediction_envs(
        predator_t,
        target_mdp,
        CombinedPolicyLearner(model1, model2, W=W),
        action_selector=MaxActionSelector(seed=123),
        env_reset=env_reset,
    )

    logp = prediction_likelihood(np.vstack(Q_estimates), np.hstack(predicted_a))

    return -logp


def fit_combined_model_learning_rate(X: Tuple[float], *args: List) -> float:
    """
    Fits a combined policy learning/value iteration model, estimating a learning rate for the policy learner.
    Intended for use with scipy optimization functions.

    Args:
        X (Tuple): Weighting parameter, learning rate.
        args (List): Other arguments. 1: Predator trajectory, 2: MDP,
        3: Subject's predicted actions, 4: Value iteration model, 5: Whether to use a generalisation kernel, 6: Whether
        to reset the model for each environment

    Returns:
        float: Log likelihood
    """

    W, alpha = X

    if np.isnan(W):
        W = 0.001
    if np.isnan(alpha):
        W = 0.001

    predator_t, target_mdp, predicted_a, model1, kernel, env_reset = args

    model2 = TDGeneralPolicyLearner(learning_rate=alpha, kernel=kernel, decay=0)

    _, Q_estimates, _, _ = action_prediction_envs(
        predator_t,
        target_mdp,
        CombinedPolicyLearner(model1, model2, W=W),
        action_selector=MaxActionSelector(seed=123),
        env_reset=env_reset,
    )

    logp = prediction_likelihood(np.vstack(Q_estimates), np.hstack(predicted_a))

    return -logp


def simulate_combined_model_learning_rate(
    X: Tuple[float], args: List
) -> Union[List, List]:
    """
    Simulates predictions using a combined policy learning/value iteration model, including a learning rate for the policy learner.
    Takes arguments in the same form as the corresponding fitting function.

    Args:
        X (Tuple): Weighting parameter, learning rate.
        args (List): Other arguments. 1: Predator trajectory, 2: MDP,
        3: Subject's predicted actions, 4: Value iteration model, 5: Whether to use a generalisation kernel, 6: Whether
        to reset the model for each environment

    Returns:
        Union[List, List]: List of predicted states, list of predicted actions
    """

    W, alpha = X

    predator_t, target_mdp, predicted_a, model1, kernel, env_reset = args

    model2 = TDGeneralPolicyLearner(learning_rate=alpha, kernel=kernel, decay=0)

    _, _, simulated_predictions, simulated_predicted_actions = action_prediction_envs(
        predator_t,
        target_mdp,
        CombinedPolicyLearner(model1, model2, W=W),
        action_selector=MaxActionSelector(seed=123),
        env_reset=env_reset,
    )

    return simulated_predictions, simulated_predicted_actions


"""
Learning models - not reset each environment (but could be), not reset for different MDPs
Goal inference model - reset for each environment, reset for different MDPs

Goal inference / learning models - goal inference reset for each environment, learning not (but could)
                                   goal inference reset for different MDPs, learning not
"""


def nan_to_zero(x: np.ndarray):
    """Removes nans and infs from an array"""
    x[np.isnan(x)] = 0
    x[np.isinf(x)] = 0
    return x


def action_prediction_envs(
    trajectories: List[List[int]],
    mdps: List[MDP],
    policy_model: BaseGeneralPolicyLearner,
    n_predictions: int = 2,
    action_selector: ActionSelector = None,
    step_reset: bool = False,
    env_reset: bool = False,
) -> Union[List, List, List]:
    """
    Estimates an agent's Q values for each action in each state of a trajectory of states, and makes predictions about its actions. This
    is run for a series of environments.

    By default, assumes a situation where a prediction is being made every 2 moves the agent makes. Q values are estimated every move,
    but predictions are based on the estimated Q values from the first state, and the returned Q values are only updated every `n_predictions`.

    Args:
        trajectories (List[List[int]]): List of state trajectories, one per environment.
        mdps (List[MDP]): List of MDPs in which the agent is acting, one per environment. Each entry can also be a list of MDPs,
        one per step in the trajectory, to allow different features at each step. If a list of MDPs is supplied, the transition
        function of the first MDP is used for all, the only thing that changes is the features.
        policy_model (BaseGeneralPolicyLearner): Model used to estimate Q values
        n_predictions (int, optional): Number of predictions to make at a time. Defaults to 2.
        action_selector (ActionSelector, optional): Action selection algorithm. Defaults to None.
        step_reset (bool, optional): If true, the policy learning algorithm is reset at each step. Defaults to False.
        env_reset (bool, optional): If true, the policy learning algorithm is reset at each environment. Defaults to False.

    Returns:
        Union[List, List, List]: Returns lists of most recent Q value estimates for each action, Q estimates at every step,
        and predicted actions at every step for each environemnt.
    """

    all_Q = []
    all_Q_estimates = []
    all_predictions = []
    all_predicted_actions = []

    # Loop through environments
    for n, trajectory in enumerate(trajectories):
        # Reset before each environment if needed
        if env_reset:
            policy_model.reset()

        Q, Q_estimates, predictions, predicted_actions = action_prediction(
            trajectory,
            mdps[n],
            policy_model,
            n_predictions,
            action_selector,
            reset=step_reset,
        )

        all_Q.append(Q)
        all_Q_estimates.append(Q_estimates)
        all_predictions.append(predictions)
        all_predicted_actions.append(predicted_actions)

    return all_Q, all_Q_estimates, all_predictions, all_predicted_actions


def action_prediction(
    trajectory: List[int],
    mdp: MDP,
    policy_model: BaseGeneralPolicyLearner,
    n_predictions: int = 2,
    action_selector: ActionSelector = None,
    reset: bool = False,
) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimates an agent's Q values for each action in each state of a trajectory of states, and makes predictions about its actions.

    By default, assumes a situation where a prediction is being made every 2 moves the agent makes. Q values are estimated every move,
    but predictions are based on the estimated Q values from the first state, and the returned Q values are only updated every `n_predictions`.

    Args:
        trajectory (List[int]): Trajectory of states
        mdp (MDP): MDP in which the agent is acting. Can also be a list of MDPs, one per step in the trajectory, to allow different
        features at each step. If a list of MDPs is supplied, the transition function of the first MDP is used for all, the only thing
        that changes is the features.
        policy_model (BaseGeneralPolicyLearner): Model used to estimate Q values
        n_predictions (int, optional): Number of predictions to make at a time. Defaults to 2.
        action_selector (ActionSelector, optional): Action selection algorithm. Defaults to MaxActionSelector().
        reset (bool, optional): If true, the policy learning algorithm is reset at each step. Defaults to False.

    Returns:
        Union[np.ndarray, np.ndarray, np.ndarray]: Returns most recent Q value estimates for each action, Q estimates at every step,
        and predicted actions at every step.
    """

    if action_selector is None:
        action_selector = MaxActionSelector(seed=123)

    if isinstance(mdp, list):
        if not len(mdp) == len(trajectory) - 1:
            raise AttributeError(
                "Must provide same number of MDPs as steps in the trajectory"
            )
    else:
        mdp = [mdp] * len(trajectory)

    # Convert trajectory to actions
    state = mdp[0]._trajectory_to_state_action(trajectory)[:, 0].astype(int)
    action = mdp[0]._trajectory_to_state_action(trajectory)[:, 1].astype(int)

    # Initial Q values
    Q = np.zeros(mdp[0].n_actions)

    # List to store estimated Q values
    Q_estimates = []

    # Predictions
    predictions = []
    predicted_state = state[0]
    predicted_actions = []

    # Fit the model to get starting Q values before any observations
    # Using an empty trajectory means learning models produce Q values of zero for all actions if they've not been fit already
    policy_model.fit(mdp[0], [[]])

    # Loop through agent moves
    for n in range(len(action)):
        # First prediction of each turn
        if n % n_predictions == 0:
            # Observed starting state
            start_state = state[n]

            # Preserve model state from first prediction to allow real model to update on every move without affecting predictions
            try:
                temp_policy_learner = policy_model.copy()
            except Exception as e:
                print(policy_model.model1.previous_mdp, policy_model.model1.q_values)
                raise e

        else:
            # Otherwise the next prediction follows from the previous predicted one
            start_state = predicted_state

        # Get Q values for this start state
        trial_Q = temp_policy_learner.get_q_values(start_state)

        # Q values for making predictions
        prediction_Q = trial_Q.copy()

        # Set impossible actions to -inf
        prediction_Q[
            [
                i
                for i in range(mdp[0].n_actions)
                if not i in mdp[0].state_valid_actions(start_state)
            ]
        ] = -np.inf

        # Add Q values to the list here - this is done before estimating as we want expected Q prior to the observation
        Q_estimates.append(prediction_Q.copy())

        # Get action
        predicted_action = action_selector.get_pi(prediction_Q[None, :])[0]
        # Get resulting state
        predicted_state = np.argmax(mdp[0].sas[start_state, predicted_action, :])

        predictions.append(predicted_state)
        predicted_actions.append(predicted_action)

        # Q VALUE ESTIMATION - after observing agent move
        if n < len(state):
            if reset:
                policy_model.reset()
            policy_model.fit(mdp[n], [state[n : n + 2]])

    Q_estimates = np.stack(Q_estimates)

    return Q, Q_estimates, predictions, predicted_actions


# TODO add tests for these functions. Please. It'll save you pain in the long run.


def BIC(n_params: int, log_lik: float, n_obs: int):
    return n_params * np.log(n_obs) - 2 * log_lik


def fit_output_to_dataframe(
    accuracy_dict: Dict,
    log_lik_dict: Dict,
    bic_dict: Dict,
    alpha_values_dict: Dict,
    w_values_dict: Dict,
    subjectID: str,
    condition: str,
):
    out = {
        "model": [],
        "accuracy": [],
        "log_lik": [],
        "BIC": [],
        "alpha_values": [],
        "w_values": [],
    }

    for model in accuracy_dict.keys():
        out["model"].append(model)
        out["accuracy"].append(accuracy_dict[model])
        out["log_lik"].append(log_lik_dict[model])
        out["BIC"].append(bic_dict[model])
        out["alpha_values"].append(alpha_values_dict[model])
        out["w_values"].append(w_values_dict[model])

    out["subjectID"] = subjectID
    out["condition"] = condition

    out = pd.DataFrame(out)

    return out


def prediction_accuracy(
    observed_predictions: np.ndarray, expected_predictions: np.ndarray
):
    """Computes the accuracy of predictions"""

    assert len(observed_predictions) == len(
        expected_predictions
    ), "Observed and expected predictions are not the same length"

    return np.equal(observed_predictions, expected_predictions).sum() / len(
        observed_predictions
    )


def get_model_fit(
    observed_predictions: List[np.ndarray],
    observed_action_predictions: List[np.ndarray],
    expected_predictions: List[np.ndarray],
    Q_estimates: List[np.ndarray],
    n_params: int = 0,
) -> Union[float, float, float]:
    """
    Calculates three model fit metrics: Prediction accuracy, log likelihood, and BIC

    Args:
        observed_predictions (List[np.ndarray]): Subject's predictions
        observed_action_predictions (List[np.ndarray]): Subjects' predicted actions
        expected_predictions (List[np.ndarray]): Model predictions
        Q_estimates (List[np.ndarray]): Estimated Q values for each action
        n_params (int, optional): Number of model parameters, used for BIC calculation. Defaults to 0.
    Returns:
        Union[float, float, float]: Returns accuracy, log likelihood, and BIC
    """

    # Stack arrays
    observed_predictions = np.hstack(observed_predictions)
    expected_predictions = np.hstack(expected_predictions)
    observed_action_predictions = np.hstack(observed_action_predictions)
    Q_estimates = np.vstack(Q_estimates)

    # Calculate
    accuracy = prediction_accuracy(observed_predictions, expected_predictions)
    log_lik = prediction_likelihood(Q_estimates, observed_action_predictions)
    bic = BIC(n_params, log_lik, len(observed_predictions))

    return accuracy, log_lik, bic


def fill_missing_predictions(pred, n_expected=20):
    new_pred = []

    for i in pred:
        if not len(i) == n_expected:
            new_pred.append(i + [-999] * (n_expected - len(i)))
        else:
            new_pred.append(i)

    return new_pred


def fit_models(
    trajectories: List[List[int]],
    mdps: List[MDP],
    predicted_states: List[List[int]],
    predicted_actions: List[List[int]],
    agent_reward_function: np.ndarray,
) -> Union[Dict, Dict, Dict, Dict, Dict, Dict]:
    """
    Fits 7 models to predictions:
    1. Policy repetition
    2. Policy learning
    3. Policy learning with generalisation
    4. Goal inference
    5. Goal inference and policy repetition
    6. Goal inference and policy learning
    7. Goal inference and policy learning with generalisation

    Args:
        trajectories (List[List[int]]): Observed agent trajectories for each environment
        mdps (List[MDP]): MDPs, one (or one list) for each environment
        predicted_states (List[List[int]]): Observed predictions for agent's moves (states)
        predicted_actions (List[List[int]]): Observed predictions for agent's moves (actions)
        agent_reward_function (np.ndarray): Agent's reward function. Assumes the agent has the same reward function across all MDPs.

    Returns:
        Union[Dict, Dict, Dict, Dict, Dict, Dict]: Returns model accuracy, log likelihood, and BIC, along with its predictions and
        fitted parameter values for learning rate and weighting parameters
    """

    # Dictionary to store outputs
    accuracy = {}
    log_lik = {}
    bic = {}
    alpha_values = {}
    w_values = {}
    model_predictions = {}

    #####################
    # POLICY REPETITION #
    #####################
    _, Q_estimates, predictions, _ = action_prediction_envs(
        trajectories,
        mdps,
        TDGeneralPolicyLearner(learning_rate=1, decay=0),
        action_selector=MaxActionSelector(seed=123),
    )

    accuracy["repetition"], log_lik["repetition"], bic["repetition"] = get_model_fit(
        predicted_states, predicted_actions, predictions, Q_estimates
    )
    alpha_values["repetition"], w_values["repetition"] = (
        np.nan,
        np.nan,
    )  # No parameters in this model
    model_predictions["repetition"] = np.array(
        fill_missing_predictions(predictions)
    ).copy()

    ###################
    # POLICY LEARNING #
    ###################

    # Estimate learning rate
    res = differential_evolution(
        fit_policy_learning,
        seed=123,
        args=(trajectories, mdps, predicted_actions, False, False),
        bounds=[(0.001, 0.999)],
    )

    # Simulate with estimated learning rate
    _, Q_estimates, predictions, _ = action_prediction_envs(
        trajectories,
        mdps,
        TDGeneralPolicyLearner(learning_rate=res.x[0], decay=0),
        action_selector=MaxActionSelector(seed=123),
    )

    (
        accuracy["policy_learning"],
        log_lik["policy_learning"],
        bic["policy_learning"],
    ) = get_model_fit(predicted_states, predicted_actions, predictions, Q_estimates, 1)

    (
        alpha_values["policy_learning"],
        w_values["policy_learning"],
    ) = (res.x[0], np.nan)
    model_predictions["policy_learning"] = np.array(
        fill_missing_predictions(predictions)
    ).copy()

    #########################
    # POLICY GENERALISATION #
    #########################

    # Estimate learning rate
    res = differential_evolution(
        fit_policy_learning,
        seed=123,
        args=[trajectories, mdps, predicted_actions, True, False],
        bounds=[(0.001, 0.999)],
    )

    _, Q_estimates, predictions, _ = action_prediction_envs(
        trajectories,
        mdps,
        TDGeneralPolicyLearner(learning_rate=res.x[0], decay=0, kernel=True),
        action_selector=MaxActionSelector(seed=123),
    )

    (
        accuracy["policy_generalisation"],
        log_lik["policy_generalisation"],
        bic["policy_generalisation"],
    ) = get_model_fit(predicted_states, predicted_actions, predictions, Q_estimates, 1)

    (
        alpha_values["policy_generalisation"],
        w_values["policy_generalisation"],
    ) = (
        res.x[0],
        np.nan,
    )
    model_predictions["policy_generalisation"] = np.array(
        fill_missing_predictions(predictions)
    ).copy()

    ##################
    # GOAL INFERENCE #
    ##################
    _, Q_estimates, predictions, _ = action_prediction_envs(
        trajectories,
        mdps,
        VIPolicyLearner(ValueIteration(), agent_reward_function),
        action_selector=MaxActionSelector(seed=123),
    )
    (
        accuracy["goal_inference"],
        log_lik["goal_inference"],
        bic["goal_inference"],
    ) = get_model_fit(predicted_states, predicted_actions, predictions, Q_estimates)
    (
        alpha_values["goal_inference"],
        w_values["goal_inference"],
    ) = (np.nan, np.nan)
    model_predictions["goal_inference"] = np.array(
        fill_missing_predictions(predictions)
    ).copy()

    ######################################
    # GOAL INFERENCE + POLICY REPETITION #
    ######################################

    # Fit VI here so it doesn't get refit constantly during parameter estimation
    VI_model = VIPolicyLearner(ValueIteration(), agent_reward_function, caching=True)
    if not isinstance(mdps[0], list):
        VI_model.fit(mdps[0], [])
    else:
        VI_model.fit(mdps[0][0], [])

    # Estimate weighting parameter
    res = differential_evolution(
        fit_combined_model,
        seed=123,
        args=[trajectories, mdps, predicted_actions, VI_model, 1, False],
        bounds=[(0, 1)],
    )

    _, Q_estimates, predictions, _ = action_prediction_envs(
        trajectories,
        mdps,
        CombinedPolicyLearner(
            VIPolicyLearner(ValueIteration(), agent_reward_function),
            TDGeneralPolicyLearner(learning_rate=1, decay=0),
            W=res.x[0],
            scale=True,
        ),
        action_selector=MaxActionSelector(seed=123),
    )
    (
        accuracy["combined_repetition"],
        log_lik["combined_repetition"],
        bic["combined_repetition"],
    ) = get_model_fit(predicted_states, predicted_actions, predictions, Q_estimates, 1)
    (
        alpha_values["combined_repetition"],
        w_values["combined_repetition"],
    ) = (
        np.nan,
        res.x[0],
    )
    model_predictions["combined_repetition"] = np.array(
        fill_missing_predictions(predictions)
    ).copy()

    #############################
    # GOAL INFERENCE + LEARNING #
    #############################

    res = differential_evolution(
        fit_combined_model_learning_rate,
        seed=123,
        args=[trajectories, mdps, predicted_actions, VI_model, False, False],
        bounds=[(0, 1), (0.001, 0.999)],
    )

    _, Q_estimates, predictions, _ = action_prediction_envs(
        trajectories,
        mdps,
        CombinedPolicyLearner(
            VIPolicyLearner(ValueIteration(), agent_reward_function),
            TDGeneralPolicyLearner(learning_rate=res.x[1], decay=0),
            W=res.x[0],
            scale=True,
        ),
        action_selector=MaxActionSelector(seed=123),
    )
    (
        accuracy["combined_learning"],
        log_lik["combined_learning"],
        bic["combined_learning"],
    ) = get_model_fit(predicted_states, predicted_actions, predictions, Q_estimates, 2)
    (
        alpha_values["combined_learning"],
        w_values["combined_learning"],
    ) = (
        res.x[1],
        res.x[0],
    )
    model_predictions["combined_learning"] = np.array(
        fill_missing_predictions(predictions)
    ).copy()

    ###################################
    # GOAL INFERENCE + GENERALISATION #
    ###################################

    res = differential_evolution(
        fit_combined_model_learning_rate,
        seed=123,
        args=[trajectories, mdps, predicted_actions, VI_model, True, False],
        bounds=[(0, 1), (0.001, 0.999)],
    )

    _, Q_estimates, predictions, _ = action_prediction_envs(
        trajectories,
        mdps,
        CombinedPolicyLearner(
            VIPolicyLearner(ValueIteration(), agent_reward_function),
            TDGeneralPolicyLearner(learning_rate=res.x[1], decay=0, kernel=True),
            W=res.x[0],
            scale=True,
        ),
        action_selector=MaxActionSelector(seed=123),
    )
    (
        accuracy["combined_generalisation"],
        log_lik["combined_generalisation"],
        bic["combined_generalisation"],
    ) = get_model_fit(predicted_states, predicted_actions, predictions, Q_estimates, 2)
    (
        alpha_values["combined_generalisation"],
        w_values["combined_generalisation"],
    ) = (
        res.x[1],
        res.x[0],
    )
    model_predictions["combined_generalisation"] = np.array(
        fill_missing_predictions(predictions)
    ).copy()

    return (
        accuracy,
        log_lik,
        bic,
        model_predictions,
        alpha_values,
        w_values,
    )


def fit_subject_predictions(
    predator_moves: pd.DataFrame,
    prey_moves: pd.DataFrame,
    predictions: pd.DataFrame,
    env_info: Dict,
    simulate: bool = False,
    simulation_options: Tuple = None,
) -> pd.DataFrame:
    """
    Fits a series of models to subjects' predictions about the predator's movements.

    Args:
        predator_moves (pd.DataFrame): A dataframe representing the states visited by the predator
        prey_moves (pd.DataFrame): A dataframe representing the states visited by the prey (i.e. the subject's moves)
        predictions (pd.DataFrame): Subject's predictions about the predator's moves
        env_info (Dict): Information about each environment
        simulate (bool): Whether to simulate predictions from a model (specified in the simulation_options argument). If True, these
        simulated predictions are used for model fitting rather than the actual subject's predictions. Defaults to False.
        simulation_options (Tuple): Options for the simulation, in a tuple of the form (model name, parameters, additional arguments).
        Model name can be either "policy", "policy_generalisation", "combined", or "combined_generalisation". Defaults to None.

    Returns:
        pd.DataFrame: A dataframe of model fitting outputs
    """

    # Make sure everything is in the right order
    predator_moves = predator_moves.sort_values(
        ["subjectID", "condition", "env", "trial", "response_number"]
    ).reset_index(drop=True)
    prey_moves = prey_moves.sort_values(
        ["subjectID", "condition", "env", "trial", "response_number"]
    ).reset_index(drop=True)
    predictions = predictions.sort_values(
        ["subjectID", "condition", "env", "trial", "response_number"]
    ).reset_index(drop=True)

    # Get simulation parameters
    if simulate:
        if simulation_options is None:
            raise ValueError("simulation_options must be specified if simulate is True")

        simulation_model, simulation_params, simulation_args = simulation_options
        if simulation_model not in [
            "policy",
            "policy_generalisation",
            "combined",
            "combined_generalisation",
        ]:
            raise ValueError("simulation_model must be either 'policy' or 'combined'")

    # Check if any data is missing
    try:
        if (
            not (predator_moves["trial"].diff() > 1).any()
            or not len(predator_moves["condition"].unique()) == 1
        ):
            # Dictionaries to store outputs
            accuracy = {}
            log_lik = {}
            bic = {}
            alpha_values = {}
            w_values = {}
            model_predictions = {}

            # Get condition that this subject is in
            condition = predator_moves["condition"].tolist()[0]

            # Get subject ID
            subject = predator_moves["subjectID"].tolist()[0]

            # Information for model fitting
            env_predator_trajectories = []
            env_mdps = []
            env_predictions = []
            env_action_predictions = []
            valid = True

            # Loop through environments
            for env in predator_moves["env"].unique():
                # Get data for this environment
                env_predator_df = predator_moves[predator_moves["env"] == env]
                env_prediction_df = predictions[predictions["env"] == env]
                env_prey_df = prey_moves[prey_moves["env"] == env]

                # Remove missing data due to getting caught
                env_predator_df = env_predator_df[env_predator_df["cellID"] != -999]
                env_prediction_df = env_prediction_df[
                    env_prediction_df["cellID"] != -999
                ]
                env_prey_df = env_prey_df[env_prey_df["cellID"] != -999]

                # Get trajectories
                predator_trajectory = [
                    env_info[condition][env].agents["Predator_1"].position
                ] + env_predator_df["cellID"].tolist()
                predicted_trajectory = env_prediction_df["cellID"].tolist()

                # Get MDP representing this environment
                mdp = env_info[condition][env].mdp

                # Ensure that we have the right number of predictions
                if len(predator_trajectory) - 1 == len(predicted_trajectory):
                    mdps = mdp

                    # GET ACTIONS FROM PREDICTED STATES
                    predicted_actions = []

                    for i in np.arange(0, len(predicted_trajectory), 2):
                        if simulation_options is None:
                            t = [
                                predator_trajectory[i],
                                *predicted_trajectory[i : i + 2],
                            ]
                            try:
                                predicted_actions += (
                                    mdp._trajectory_to_state_action(t)[:, 1]
                                    .astype(int)
                                    .tolist()
                                )
                            except Exception as e:
                                print("Subject = ", subject)
                                print("Environment = ", env)
                                print("Trajectory = ", t)
                                print("Entire trajectory = ", predator_trajectory)
                                print("Move = ", i)
                                raise e
                        else:
                            predicted_actions = [0] * (len(predicted_trajectory) - 1)

                    # Add info for this environment to list
                    env_predator_trajectories.append(predator_trajectory)
                    env_mdps.append(mdps)

                    if not simulate:
                        env_predictions.append(predicted_trajectory)
                        env_action_predictions.append(predicted_actions)

                # Otherwise skip this subject
                else:
                    print(len(predator_trajectory), len(predicted_trajectory))
                    raise ValueError(
                        "Subject {0}, env {1}, cond {2} has mismatch between predicted and observed moves".format(
                            subject, env, condition
                        )
                    )

            if valid:
                if simulate:
                    # Simulate model predictions
                    if simulation_model == "policy":
                        (
                            env_predictions,
                            env_action_predictions,
                        ) = simulate_policy_learning(
                            simulation_params,
                            (env_predator_trajectories, env_mdps, [], False, False),
                        )
                    elif simulation_model == "policy_generalisation":
                        (
                            env_predictions,
                            env_action_predictions,
                        ) = simulate_policy_learning(
                            simulation_params,
                            (env_predator_trajectories, env_mdps, [], True, False),
                        )
                    elif "combined" in simulation_model:
                        if "generalisation" in simulation_model:
                            generalisation = True
                        else:
                            generalisation = False

                        # Fit VI here so it doesn't get refit constantly during parameter estimation
                        VI_model = VIPolicyLearner(
                            ValueIteration(),
                            env_info[condition][env]
                            .agents["Predator_1"]
                            .reward_function,
                            caching=True,
                        )
                        if not isinstance(env_mdps[0], list):
                            VI_model.fit(env_mdps[0], [])
                        else:
                            VI_model.fit(env_mdps[0][0], [])

                        (
                            env_predictions,
                            env_action_predictions,
                        ) = simulate_combined_model_learning_rate(
                            simulation_params,
                            (
                                env_predator_trajectories,
                                env_mdps,
                                [],
                                VI_model,
                                generalisation,
                                False,
                            ),
                        )

                (
                    accuracy,
                    log_lik,
                    bic,
                    model_predictions,
                    alpha_values,
                    w_values,
                ) = fit_models(
                    env_predator_trajectories,
                    env_mdps,
                    env_predictions,
                    env_action_predictions,
                    env_info[condition][env].agents["Predator_1"].reward_function,
                )

                out_df = fit_output_to_dataframe(
                    accuracy,
                    log_lik,
                    bic,
                    alpha_values,
                    w_values,
                    subject,
                    condition,
                )

                return out_df, model_predictions

        else:
            print("Missing data, skipping")
    except Exception as e:
        print("PROBLEM WITH DATA")
        raise e


# Taken from https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697#58936697
@contextlib.contextmanager
def tqdm_joblib(tqdm_object: tqdm):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument

    From https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697#58936697
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def fit_prediction_models(
    predator_moves: pd.DataFrame,
    prey_moves: pd.DataFrame,
    predictions: pd.DataFrame,
    env_info: Dict,
    n_jobs: int = 1,
    simulation_options: List[Tuple] = None,
) -> Union[pd.DataFrame, List]:
    """
    Fits 7 different prediction models to subjects' data:

    1. Policy repetition
    2. Policy learning
    3. Policy learning with generalisation
    4. Goal inference
    5. Goal inference and policy repetition
    6. Goal inference and policy learning
    7. Goal inference and policy learning with generalisation

    Can also simulate data from these models and then fit the model to this simulated data to facilitate
    parameter/model recovery tests.

    Args:
        predator_moves (pd.DataFrame): Dataframe containing moves made by the predator
        prey_moves (pd.DataFrame): Dataframes containing moves made my prey
        predictions (pd.DataFrame): Dataframe containing predictions made by subject
        env_info (Dict): Environment information
        n_jobs (int, optional): Number of jobs for parallel processing. Defaults to 1.
        simulation_options (List[Tuple]): If provided, simulated predictions are used for model fitting rather than the
        actual subject's predictions. Options for the simulation are provided as a list of tuples of the form
        (model name, parameters, additional arguments). The list should contain one entry per subject in the dataframes.
        Model name can be either "policy", "policy_generalisation", "combined", or "combined_generalisation". Defaults to None.

    Returns:
        Union[pd.Dataframe, List]: Returns fit statistics and predictions made by the models
    """

    if simulation_options is not None:
        simulate = True
    else:
        simulate = False
        simulation_options = [None] * len(predator_moves["subjectID"].unique())

    subjects = list(predator_moves["subjectID"].unique())

    if simulate and not len(simulation_options) == len(subjects):
        raise ValueError("Number of subjects must match number of simulation options")

    fit_dfs = []
    model_prediction_list = []

    if n_jobs == 1:
        for n, sub in enumerate(tqdm(subjects)):
            fit, model_predictions = fit_subject_predictions(
                predator_moves[predator_moves["subjectID"] == sub],
                prey_moves[prey_moves["subjectID"] == sub],
                predictions[predictions["subjectID"] == sub],
                env_info,
                simulate=simulate,
                simulation_options=simulation_options[n],
            )

            fit_dfs.append(fit)
            model_prediction_list.append(model_predictions)

    else:
        from joblib import Parallel, delayed

        print("Fitting in parallel, {0} jobs".format(n_jobs))
        with tqdm_joblib(tqdm(desc="Evaluation", total=len(subjects))):
            fits = Parallel(n_jobs=n_jobs)(
                delayed(fit_subject_predictions)(
                    predator_moves[predator_moves["subjectID"] == sub],
                    prey_moves[prey_moves["subjectID"] == sub],
                    predictions[predictions["subjectID"] == sub],
                    env_info,
                    simulate=simulate,
                    simulation_options=simulation_options[n],
                )
                for n, sub in enumerate(subjects)
            )

        fit_dfs = [i[0] for i in fits]

    all_subject_fit = pd.concat(fit_dfs)

    return all_subject_fit, model_prediction_list
