# coding=utf-8
#
# The copyright of this file belongs to Feedzai. The file cannot be
# reproduced in whole or in part, stored in a retrieval system,
# transmitted in any form, or by any means electronic, mechanical,
# photocopying, or otherwise, without the prior permission of the owner.
#
# (c) 2020 Feedzai, Strictly Confidential
import logging
import numpy as np


def get_one_hot_encoding(a):
    """Create one hot representation of provided array.

    Parameters
    ----------
    a: array-like with shape (n_samples, 1)
        Array to be converted into one hot encoding.

    Return
    ------
    array-like with shape (n_samples, n_a_values)
        The one hot encoded matrix. 

    dict
        The mapping feature value -> column index of the created 
        representation.
    """
    assert len(a.shape) == 1 or a.shape[1] == 1, f"Invalid value array with shape: {a.shape}"

    if a.dtype == float:
        logging.warn("Converting floating point values in `a` to ints")
        a = a.astype(int)

    n_samples = a.size()
    a_values = np.unique(a)
    n_values = len(a_values)

    # Map a_values to ix if not int
    # E.g., a = ["a", "b", "a", "c"] would be mapped to => [0, 1, 0, 2]
    if a.dtype != int:
        for ix, val in enumerate(a_values):
            a[a == val] = ix

    # One hot encoded matrix
    A = np.zeros(n_samples, n_values)
    A[np.arange(n_samples), a] = 1
    # Note:
    # Assuming the attribute values are exclusively, the absence of n-1
    # feature values, implied the other feature value is true.
    mapping = {val: i for i, val in enumerate(a_values[:-1])}
    # Having n attribute values, we can represent them in terms of n-1 
    mapping.update({a_values[-1]: a_values[:-1]})
    return A[:,:-1], mapping