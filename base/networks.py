# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved. #
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
from torch import nn
from tslearn.metrics import dtw_path as tsldtw_path

from base.utils import md_center


class proto_clser(object):
    """
    Classifier with class-specific prototype sequences.
    """

    def __init__(self, train_loader, Train_N, Class_N, DTW_w_p):
        self.train_loader = train_loader
        self.Train_N = Train_N
        self.Class_N = Class_N
        self.DTW_w_p = DTW_w_p

    def data_of_class(self):
        """
        Get class speficied data mat
        """
        class_data_dict = {}
        for step, batch_data in enumerate(self.train_loader):
            seqs_tsor = batch_data["data"]
            labels_tsor = batch_data["label"]
            seqs_np = seqs_tsor.numpy()
            labels = labels_tsor.numpy().tolist()
            bz = len(labels)
            assert bz == seqs_np.shape[0], (bz, seqs_np.shape)
            for i in range(bz):
                label = labels[i]
                seq = seqs_np[i]
                if label in class_data_dict:
                    class_data_dict[label].append(seq)
                else:
                    class_data_dict[label] = [seq]
        assert len(class_data_dict) == self.Class_N, (len(class_data_dict), self.Class_N)
        return class_data_dict

    def get_init_kernels(self):
        """
        Get the class-specific medoid sequences.
        """
        class_data_dict = self.data_of_class()
        init_kernels_dict = {}
        for c in range(self.Class_N):
            arr_list = class_data_dict[c]
            arr_mat = np.stack(arr_list)
            init_kernels_dict[c] = torch.from_numpy(md_center(arr_mat)).detach()
        seq_t_len = arr_mat.shape[1]
        return init_kernels_dict, seq_t_len

    def init_clser_kernels(self, init_kernels_dict, seq_t_len):
        """
        Build DTW classifier;
        And Init. the classifier kernels with medoid sequences.
        """
        kernel_len = seq_t_len
        # DTW window constraint
        if self.DTW_w_p == 1:
            DTW_BW = None
        else:
            DTW_BW = int(seq_t_len * self.DTW_w_p) + 1

        dtw_list = []
        for c in range(self.Class_N):
            c_kernel = init_kernels_dict[c].type(torch.FloatTensor)
            # Class-specific DTW kernel/prototype with medoid seq. as Init.
            dtw_list.append(DTWKERNEL(kernel_len, seq_t_len, _kernel=c_kernel))
        assert len(dtw_list) == self.Class_N, (len(dtw_list), self.Class_N)
        # Warp the kernels/prototypes as a layer.
        dtwlayer_list = [DTWLAYER(dtw_list, DTW_BW=DTW_BW)]
        # Warp such DTWLAYER as classifier.
        clser = DTW_CLSER(dtwlayer_list, self.Class_N)
        return clser

    def build(self):
        """
        Build DTW classifier in Two Steps:
        1. Find the class-specified medoid sequences;
        2. Init. the prototypes/kernels in classifier with medoid sequences.
        """
        init_kernels_dict, seq_t_len = self.get_init_kernels()
        clser = self.init_clser_kernels(init_kernels_dict, seq_t_len)
        return clser


class DTWKERNEL(nn.Module):
    """
    Define a DTW Kernel.
    """

    def __init__(self, _kernel_len, _input_len, **kwargs):
        super(DTWKERNEL, self).__init__()
        self.kernel_len = _kernel_len
        self.input_len = _input_len
        if "_kernel" not in kwargs:
            raise ValueError("random init kernel will be given as well.")
        else:
            self.kernel = nn.Parameter(kwargs.get("_kernel"))
        self.out_len = 1

    def compute(self, x, band_width=None):
        """
        DTW is used to find the optimal alignment path;
        Compute the discrepancy value between the input and kernel according to optimal path.
        """
        if band_width is None:
            path, dist = tsldtw_path(self.kernel.detach().cpu().numpy(), x.detach().cpu().numpy())
        else:
            path, dist = tsldtw_path(
                self.kernel.detach().cpu().numpy(), x.detach().cpu().numpy(), sakoe_chiba_radius=band_width
            )
        k_list, i_list = [], []
        for i in range(len(path)):
            k_list.append(path[i][0])
            i_list.append(path[i][1])
        return torch.sum((self.kernel[k_list] - x[i_list]).pow(2))

    def forward(self, x, band_width=None):
        """
        Compute DTW distance between an input series with the kernel/prototype
        """
        return self.compute(x, band_width=band_width)


class DTWLAYER(nn.Module):
    """
    Given a list of kernels (_dtw_list), warp them in the DTW layer;
    Given sequences, return their discrepancy values with all kernels.
    """

    def __init__(self, _dtw_list, DTW_BW=None, avg_t_len=False):
        super(DTWLAYER, self).__init__()
        self.num_filter = len(_dtw_list)
        self.filters = nn.ModuleList(_dtw_list)
        self.filter_outlen = _dtw_list[0].out_len
        self.out_len = self.num_filter * self.filter_outlen
        self.DTW_BW = DTW_BW
        self.avg_t_len = avg_t_len

    def forward_one_batch(self, x):
        """
        For an input seq, compute discrenapcy with each kernel;
        Aggregate the discrenapcy values as the output of a sample.
        """
        out_list = []
        assert len(x.shape) == 1, x.shape
        in_t_len = float(x.shape[0])
        for i in range(self.num_filter):
            if self.avg_t_len:
                out_list.append(self.filters[i].forward(x, band_width=self.DTW_BW) / in_t_len)
            else:
                out_list.append(self.filters[i].forward(x, band_width=self.DTW_BW))

        out = torch.stack(out_list)
        assert out.shape[0] == self.num_filter, (out.shape, (self.num_filter, self.filter_outlen))
        return out

    def forward(self, x):
        """
        Forwarding each input sequence for DTW computation with all kernels.
        Aggregate the outputs in a mini-batch.
        """
        out_list = []
        for k in range(x.shape[0]):
            out_list.append(self.forward_one_batch(x[k]))
        out = torch.stack(out_list)
        assert out.shape[0] == x.shape[0], (out.shape, (x.shape[0], self.num_filter, self.filter_outlen))
        return out


class DTW_CLSER(nn.Module):
    """
    Using a DTWLAYER as the classifier.
    """

    def __init__(self, _dtwlayer_list, Class_N):
        super(DTW_CLSER, self).__init__()
        assert len(_dtwlayer_list) == 1, len(_dtwlayer_list)
        self.dtwlayers = nn.ModuleList(_dtwlayer_list)
        assert Class_N > 0, Class_N
        self.Class_N = Class_N

    def forward(self, x):
        # DTWLAYER computes the discrepancies/distances of inputs against the class-specific kernels/prototypes.
        assert len(x.shape) == 2, x.shape
        DTW_dist = self.dtwlayers[-1](x)
        sample_N = x.shape[0]

        DTW_score = -1.0 * DTW_dist
        assert len(DTW_score.shape) == 2, DTW_score.shape
        assert DTW_score.shape[0] == sample_N, DTW_score.shape
        assert DTW_score.shape[1] == self.Class_N, DTW_score.shape

        return DTW_dist, DTW_score
