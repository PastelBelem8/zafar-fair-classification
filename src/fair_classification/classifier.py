# coding=utf-8
#
# The copyright of this file belongs to Feedzai. The file cannot be
# reproduced in whole or in part, stored in a retrieval system,
# transmitted in any form, or by any means electronic, mechanical,
# photocopying, or otherwise, without the prior permission of the owner.
#
# (c) 2020 Feedzai, Strictly Confidential
from typing import Any, Dict, Iterable

from abc import ABC, abstractmethod
import cvxpy as cvx
import dccp
import random
import numpy as np


class LinearClassifier(ABC):
    def __init__(self, n_features, init, reg_term=0):
        self.n_weights = n_features + 1
        self.weights_ = cvx.Variable(n_features + 1)
        self.weights_.value = np.hstack(([1], init))
        self.regularization = reg_term

    @property
    def coeffs(self):
        return self.weights_.value

    @staticmethod
    def _add_intercept(X):
        intercepts = np.ones((X.shape[0], 1))
        return np.hstack((intercepts, X))

    @abstractmethod
    def _loss(self, X, y):
        raise NotImplementedError("Subclasses must override this method")
    
    def loss(self, X, y):
        X = LinearClassifier._add_intercept(X)
        total_loss = 0
        if 0 < self.regularization <= 1:
            total_loss += cvx.sum_squares(self.weights_[1:]) * self.regularization

        total_loss += self._loss(X, y)
        return total_loss
        
    def decision_boundary(self, X, y):
        X = LinearClassifier._add_intercept(X)
        return cvx.multiply(y, X * self.weights_)

    def predict_proba(self, X):
        X = LinearClassifier._add_intercept(X)
        return X @ self.coeffs


class LogisticRegression(LinearClassifier):
    def _loss(self, X, y):
        # logistic = log (1 + e^z)
        logloss = cvx.logistic(cvx.multiply(-y, X * self.weights_))
        return logloss
    # obj += cvx.sum(  cvx.logistic( cvx.multiply(-y_k, X_k*w[k]) )  ) / num_all
    # notice that we are dividing by the length of the whole dataset, and not
    # just of this sensitive group. this way, the group that has more people
    # contributes more to the loss


class SVMLinear(LinearClassifier):
    def _loss(self, X, y):
        X = LinearClassifier._add_intercept(X)
        svm = 1 - cvx.multiply(y,  X * self.weights_)
        hinge_loss = cvx.maximum(0, svm)
        return hinge_loss


class ConstrainedClassifier:
    """A constrained classifier.

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

    classifier : Classifier, default=LogisticRegression
        The loss function to optimize. Currently supported version
        is LogisticRegression, which corresponds to the
        log loss.

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

    classifier: estimator class
        The classifier class.

    tolerance: float
        The absolute accuracy of the solution.

    constraints_: the constraint
        The ``DCCP`` constraints.

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

    def __init__(
        self,
        constraints_configs: Dict[str, Any] = {},
        max_iter: int = 100,
        classifier = LogisticRegression,
        tol: float = 1e-8,
        random_state=None
    ):
        self.rand = np.random.RandomState(
            random_state) if random_state else np.random.RandomState()

        self.tolerance = tol
        self.max_iter = max_iter
        self.constraints_configs = constraints_configs
        self.constraints_type = constraints_configs.get('type', 0)

        # DCCP parameters
        self.dccp_tau = constraints_configs.get('tau', 0)
        self.dccp_mu = constraints_configs.get('mu', 0)
        self.dccp_max_iter = constraints_configs.get('max_iter', 50)

        self.dccp_params = {
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
        self.hotstart_unconstrained = constraints_configs.get(
            'hotstart_unconstrained', True)

        self.estimator_cls = classifier
        self.estimator_ = None
        self.constraints_ = self._init_constraints( # TODO - Parse constraint configs
            X, y, X_protected, self.covariance_thresholds,
            self.constraints_type, self.weights)

    def fit(self, X, y, X_protected: Dict[str, np.ndarray] = None, sample_weight=None):
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
        n_samples, n_features = X.shape
        estimator = self.estimator_cls(n_features, self.rand.random(n_features))

        if self.hotstart_unconstrained:
            logging.info("Bootstrapping constr. problem w/ unconstrained solution")
            loss = estimator.loss(X, y)
            unc_problem = cvx.Problem(cvx.Minimize(cvx.sum(loss) / n_samples), [])
            unc_problem.solve()  # the solution is the weights (which got updated)
            logging.info(f"Unconstrained solution: {estimator.coeffs}")

        # Constrained Problem
        loss = estimator.loss(X, y)
        problem = cvx.Problem(cvx.Minimize(cvx.sum(loss) / n_samples)), self.constraints_)
        logging.debug(
            f"Problem is DCP (disciplined convex program): {problem.is_dcp()}")
        logging.debug(
            f"Problem is DCCP (disciplined convex-concave program): {dccp.is_dccp(problem)}")

        problem.solve(
            tau = self.dccp_tau,
            mu = self.dccp_mu,
            max_iters = self.max_iter,
            max_iter = self.max_iter_dccp,
            **self.dccp_params,
        )
        assert(problem.status is None or problem.status ==
               "Converged" or problem.status == "optimal")
        logging.info(f"Optimization done, problem status: {problem.status}")

        self.estimator_ = estimator
        return self

    def predict_proba(self, X):
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
        return self.estimator_.predict_proba(X).flatten()
