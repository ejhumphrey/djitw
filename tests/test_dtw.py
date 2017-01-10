import pytest

import numpy as np

import djitw


def test_band_mask():
    mask = np.zeros((8, 8), dtype=np.bool)
    djitw.band_mask(.25, mask)
    exp = np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1]])
    np.testing.assert_array_equal(exp, mask.astype(int))

    mask = np.zeros((8, 12), dtype=np.bool)
    djitw.band_mask(.25, mask)

    exp = np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])

    np.testing.assert_array_equal(exp, mask.astype(int))
