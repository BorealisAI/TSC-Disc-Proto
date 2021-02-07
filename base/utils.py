# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved. #
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
from tslearn.metrics import dtw as tsldtw


def z_norm_1d(seq_1d_arr):
    """
    z norm a sequence.
    """
    assert len(seq_1d_arr.shape) == 1, seq_1d_arr.shape
    mean = np.mean(seq_1d_arr)
    std = np.std(seq_1d_arr)
    return (seq_1d_arr - mean) / (std + 1e-8)


def z_norm_seqs(seqs_mat):
    """
    z norm 1d sequences. seqs_mat in shape [N, T]
    """
    assert len(seqs_mat.shape) == 2, seqs_mat.shape
    N = seqs_mat.shape[0]
    normed_arrs = []
    for i in range(N):
        normed_arrs.append(z_norm_1d(seqs_mat[i]))
    assert len(normed_arrs) == N, (len(normed_arrs), N)
    return np.stack(normed_arrs)


class np2tensor(object):
    def __call__(self, np_arr):
        return torch.from_numpy(np_arr)


def isnan(x):
    return x != x


def md_center(seqs):
    """
    Find the medoid sequence and return.
    """
    N = len(seqs)
    score_mat = np.zeros((N, N))
    for i in range(N):
        s = seqs[i]
        for j in range(N):
            t = seqs[j]
            if i < j:
                score_mat[i, j] = tsldtw(s, t)
    score_mat = score_mat + score_mat.transpose()
    m_idx = np.argmin(np.sum(score_mat, axis=0))
    return seqs[m_idx]
