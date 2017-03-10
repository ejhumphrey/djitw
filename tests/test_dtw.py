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


def test_dtw_core():
    pass


def test_dtw_core_masked():
    pass


def test_dtw():
    dist_mat = np.ones([8, 5])
    x_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    y_idx = np.array([0, 1, 1, 2, 2, 3, 3, 4])
    dist_mat[x_idx, y_idx] = 0.0
    print(dist_mat)
    x_out, y_out, score = djitw.dtw(dist_mat)
    np.testing.assert_array_equal(x_idx, x_out)
    np.testing.assert_array_equal(y_idx, y_out)
    assert score < 1
