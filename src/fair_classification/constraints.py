# coding=utf-8
#
# The copyright of this file belongs to Feedzai. The file cannot be
# reproduced in whole or in part, stored in a retrieval system,
# transmitted in any form, or by any means electronic, mechanical,
# photocopying, or otherwise, without the prior permission of the owner.
#
# (c) 2020 Feedzai, Strictly Confidential
from abc import ABC, abstractmethod
from .utils import get_one_hot_encoding

import cvxpy as cvx
import numpy as np


class Constraint(ABC):
    """Constraints."""
    def __init__(self, label_pos=+1, label_neg=-1):
        self.label_neg = label_neg
        self.label_pos = label_pos
        self.constraints_ = []

    @abstractmethod
    def evaluate(self, X, y, A, f, threshold):
        """Evaluate the constraint.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_dims)
            Training sample.

        y: array-like of shape (n_samples, 1)
            Ground truth label of the training sample.

        A: array-like of shape (n_samples, 1) or (n_samples,)
            Sensitive Attribute of the training sample.

        f: calllable
            The learner's prediction. Should reflect how far a given 
            instance is from the margin (i.e., decision boundary). In
            other words, it should yield the signed distance from the 
            feature vector, x, to the decision boundary.

        threshold:
            The fairness threshold to consider when creating the constraints.
            In the original code, this threshold was usually set as the absolute
            difference between two different groups for a given constraint.
            Implementation wise, this was previously defined as  ``abs(threshold[1]
            - threshold[0])`` and the user was forced to define thresholds for the 
            two attribute values.

        Returns
        -------
        iterable
            The list of DCCP constraints
        """
        raise NotImplementedError("Subclasses must override this method")


class MisclassificationConstraint(Constraint):
    """Mitigate misclassification rate."""
    def evaluate(self, X, y, A, f, threshold):
        A, enc_map = get_one_hot_encoding(A)

        if len(enc_map) == 2: # is_binary
            A_values = np.unique(A)
            instances_per_val = {val: sum(A == val) for val in A_values}

            constraints_per_val = {}
            for val in A_values:
                idx = X[A] == val

                # DCCP constraints setup
                # 1. Compute the distance from the boundary (for the negative labels)
                # 2. Compute the base_rate per attribute value (i.e., the Ni/N)
                # 3. Compute the total error amongst all FPRs (through the sum)
                #    This error will be negative, because FP := {y = -1, f(x) > 0}
                dist_to_bound = f(X[idx], y[idx])  # g(y, x)
                base_rate_nval = instances_per_val[val] / len(A)

                total_error = cvx.sum(cvx.minimum(0, dist_to_bound)) 
                constraint_val[val] = base_rate_nval * total_error # avg misclassification distance from boundary

            # DCCP constraints
            self.constraints_ = [
                constraints_per_val[1] <= constraints_per_val[0] + threshold,
                constraints_per_val[1] >= constraints_per_val[0] - threshold,
            ]
        else:
            raise NotImplementedError("Categorical attributes not supported")
        return self.constraints_


class FPRConstraint(Constraint):
    """Mitigate FPR imbalance across groups."""
    def evaluate(self, X, y, A, f, threshold):
        ln = (y == self.label_neg)
        X, y, A = X[ln], y[ln], A[ln]
        A, enc_map = get_one_hot_encoding(A)

        if len(enc_map) == 2: # is_binary
            A_values = np.unique(A)
            fp_per_val = {val: sum(A == val) for val in A_values}

            ln = sum(ln)
            constraints_per_val = {}
            for val in A_values:
                idx = X[A] == val

                # DCCP constraints setup
                # 1. Compute the distance from the boundary (for the negative labels)
                # 2. Compute the base_rate per attribute value (i.e., the Ni/N)
                # 3. Compute the total error amongst all FPRs (through the sum)
                #    This error will be negative, because FP := {y = -1, f(x) > 0}
                dist_to_bound = cvx.multiply(
                    y[idx], f(X[idx]))  # g(y, x) = y.f(x)
                base_rate = fp_per_val[val] / ln

                total_fp_error = cvx.sum(cvx.minimum(0, dist_to_bound))
                constraint_val[val] = base_rate * total_fp_error

            # DCCP constraints
            self.constraints_ = [
                constraints_per_val[1] <= constraints_per_val[0] + threshold,
                constraints_per_val[1] >= constraints_per_val[0] - threshold,
            ]
        else:
            raise NotImplementedError("Categorical attributes not supported")
        return self.constraints_


class FNRConstraint(Constraint):
    """Mitigate FNR imbalance"""
    def evaluate(self, X, y, A, f, threshold):
        lp = (y == self.label_pos)
        X, y, A = X[lp], y[lp], A[lp]
        A, enc_map = get_one_hot_encoding(A)

        if len(enc_map) == 2: # is_binary
            A_values = np.unique(A)
            fn_per_val = {val: sum(A == val) for val in A_values}

            lp = sum(lp)
            constraints_per_val = {}
            for val in A_values:
                idx = X[A] == val

                # DCCP constraints setup
                # 1. Compute the distance from the boundary (for the negative labels)
                # 2. Compute the base_rate per attribute value (i.e., the Ni/N)
                # 3. Compute the total error amongst all FNRs (through the sum)
                #    This error will be negative, because FN := {y = 1, f(x) < 0}
                dist_to_bound = cvx.multiply(
                    y[idx], f(X[idx]))  # g(y, x) = y.f(x)
                base_rate = fn_per_val[val] / lp

                total_fn_error = cvx.sum(cvx.minimum(0, dist_to_bound))
                constraint_val[val] = base_rate * total_fn_error

            # DCCP constraints
            self.constraints_ = [
                constraints_per_val[1] <= constraints_per_val[0] + threshold,
                constraints_per_val[1] >= constraints_per_val[0] - threshold,
            ]
        else:
            raise NotImplementedError("Categorical attributes not supported")
        return self.constraints_


class EqualOddsConstraint(Constraint):
    """Simultaneously mitigate imbalances in FPR and FNR."""
    def evaluate(self, X, y, A, f, fpr_threshold, fnr_threshold):
        self.constraints_ = []
        self.constraints_.extend(FPRConstraint.evaluate(X, y, A, f, fpr_threshold))
        self.constraints_.extend(FNRConstraint.evaluate(X, y, A, f, fnr_threshold))
        return self.constraints_