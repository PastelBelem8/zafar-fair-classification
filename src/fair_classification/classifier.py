# coding=utf-8
#
# The copyright of this file belongs to Feedzai. The file cannot be
# reproduced in whole or in part, stored in a retrieval system,
# transmitted in any form, or by any means electronic, mechanical,
# photocopying, or otherwise, without the prior permission of the owner.
#
# (c) 2020 Feedzai, Strictly Confidential
from typing import Any, Dict, Iterable

import enum
import cvxpy as cvx
import dccp
import random
import sys
import numpy as np


# TODO
# - Create general class which abstracts Convex-Concave problems?

class CovarianceThreshold:
    """# note thresh for constraints will be cov_thresh[1] - cov_thresh[0]
# had trouble satisfying exact 0s here in synthetic data tests, so go with small epsilon,
# 1e-6...
sensitive_attrs_to_cov_thresh =
{"race_nonwhite": {0:{0:0, 1:1e-6}, 1:{0:0, 1:1e-6}, 2:{0:0, 1:1e-6}}}
# zero covariance threshold, means try to get the fairest solution
     x_control: dictionary of the type {"s": [...]}, key "s" is the sensitive
      feature name, and the value is a 1-d list with n
       elements holding the sensitive feature values
    """

    def __init__(self):
        self.attr_configs={}
        self.attr_type={}

    def set_attr(self, attr_name: str, attr_values: Iterable):
        self.attr_configs[attr_name]=list(attr_values)
        self.attr_type[attr_name]="categorical" if len(
            attr_type) > 2 else "binary"

    def set_threshold(self, constraint: Constraint, attr_name: str, **threshold_per_attr):
        pass


class FairConstraint:
    """A linear constraint classifier.

    A fairness constrained classifier [1] is a  ...
    trains the model subject to various fairness constraints.
    If no constraints are given, then simply trains an unaltered classifier.

    Parameters:
    ----------
    constraints_configs : <str, Any> dict, default=None
        Configurations of the fairness constraints.

    max_iter : int, default=100
        Maximum number of iterations taken for the ``DCCP`` solver
        to converge.

    loss : str, default='logreg'
        The loss function to optimize. Currently supported version
        is 'logreg', which corresponds to the logistic loss problem.

    tol : float, default=1e-8
        The absolute accuracy of the solution. Corresponds to the
        parameter ``abstol`` in the cvx package.

    random_state : int or RandomState, default=None
        Sample weights. If None, the sample weights are initialized to
        ``1 / n_samples``.

    Attribute
    -------
    rand : pseudo-generator
        The pseudo-random number generator.

    loss_fn: str
        The name of the loss function.

    tolerance: float
        The absolute accuracy of the solution.

    constraint_type: str with the constraint type
        Name of the constraint type specified.

    constraints_configs: dict of constraint configs
        The constraint configurations.

    dccp_tau: float
        The ``DCCP`` framework parameter to control the
        relative importance of the constraints.

    dccp_mu: float
        The ``DCCP`` framework parameter to control the degree
        by which ``dccp_tau`` increases every iteration.

    dccp_max_iter: int
        The ``DCCP`` framework parameter to control the number
        of iterations to run.

    covariance_thresholds: list of dict with covariance thresholds
        The dict of features with covariance thresholds for each constraint
        type. To define threshold for FPR constraint type specify
        ``{"FPR": <threshold_value>}``.

    hotstart_unconstrained: boolean
        Whether to use the best solution in the corresponding
        unconstrained problem to hotstart the constrained one.
        If false, any feasible solution will be used.

    References
    ----------
    .. [1] M. B. Zafar, I. Valera, M. G. Rodriguez, K. P. Gummadi,
        "Fairness Constraints: Mechanisms for Fair Classification", AISTATS 2017.
    .. [2] M. B. Zafar, I. Valera, M. G. Rodriguez, K. P. Gummadi,
        "Fairness Beyond Disparate Treatment & Disparate Impact: Learning
         Classification without Disparate Mistreatment", WWW 2017.
    .. [3] M. B. Zafar, I. Valera, M. G. Rodriguez, K. P. Gummadi, A. Weller,
        "From Parity to Preference-based Notions of Fairness in Classification",
         NeurIPS 2017.

    Examples
    --------
    >>> 
    """

    def __init__(self, constraints_configs: Dict[str, Any]={}, max_iter: int=100, loss: str='logreg', tol: float=1e-8, random_state=None):
        self.rand=np.random.RandomState(
            random_state) if random_state else np.random.RandomState()

        self.loss_fn=loss
        self.tolerance=tol
        self.max_iter=max_iter
        self.constraints_configs=constraints_configs
        self.constraints_type=constraints_configs.get('type', 0)

        # DCCP parameters
        self.dccp_tau=constraints_configs.get('tau', 0)
        self.dccp_mu=constraints_configs.get('mu', 0)
        self.dccp_max_iter=constraints_configs.get('max_iter', 50)

        self.dccp_params={
                "tau_max": 1e10,
                "solver": cvx.ECOS,
                "verbose": False,
                "feastol": tol,
                "abstol": tol,
                "reltol": tol,
                "feastol_inacc": tol,
                "abstol_inacc": tol,
                "reltol_inacc": tol,
        }
        self.dccp_params.update(constraints_configs.get("dccp_params", {}))

        # Constraint Optimization parameters
        self.covariance_thresholds=constraints_configs.get(
            'covariance_thresholds', True)
        self.hotstart_unconstrained=constraints_configs.get(
            'hotstart_unconstrained', True)

        self.weights_=None
        self.constraints_=None

    def fit(self, X, y, X_protected: Dict[str, np.ndarray]=None, sample_weight=None):
        """Build a fair classifier with fairness constraints from the training set (X, y).

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        X_protected : <str, array-like of shape (n_samples,)> dict,
            default=None
            Sample protected attributes. If None, ....

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, the sample weights are initialized to
            ``1 / n_samples``.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # TODO --> Fit Intercept (add column w/ 1s)
        n_samples, n_features=X.shape
        self.weights_=cvx.Variable(n_features)
        self.weights_.value=self.rand.random(n_features)

        self.constraints_=self.init_constraints(
            X, y, X_protected, self.covariance_thresholds, self.constraints_type, self.weights)

        # TODO - Refactor this
        if self.loss_fn == "logreg":
            # constructing the logistic loss problem
            # we are converting y to a diagonal matrix for consistent

            def logistic_reg(X, y, w):
                logistic=cvx.logistic(cvx.multiply(-y, X*self.weights_))
                return cvx.sum(logistic) / n_samples
        else:
            raise NotImplementedError(
                f"Unavailable loss function: {self.loss_fn}")

        if self.hotstart_unconstrained:
            p=cvx.Problem(cvx.Minimize(loss), [])
            p.solve()  # the solution is the weights (which got updated)

        # Constrained Problem
        problem=cvx.Problem(cvx.Minimize(loss), self.constraints_)
        print("Problem is DCP (disciplined convex program):", problem.is_dcp())
        print("Problem is DCCP (disciplined convex-concave program):",
              dccp.is_dccp(problem))

        problem.solve(
            tau=self.dccp_tau,
            mu=self.dccp_mu,
            max_iters=self.max_iter,
            max_iter=self.max_iter_dccp,
            **self.dccp_params,
        )
        assert(problem.status is None or problem.status ==
                "Converged" or problem.status == "optimal")
        print("Optimization done, problem status:", problem.status)

        # check that the fairness constraint is satisfied
        for constraint in self.constraints_:
            # can comment this out if the solver fails too often,
            # but make sure that the constraints are satisfied empirically.
            # alternatively, consider increasing tau parameter
            assert(constraint.value() == True)

        return self

    def predict_proba(self, X, threshold=0):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape [n_samples]
            Predicted class label per sample.
        """
        n_features=self.weights_.value.shape[1]
        if X.shape[1] + 1 != n_features:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (X.shape[1], n_features - 1))

        scores=X @ np.array(self.weights_.value).flatten()
        return scores
