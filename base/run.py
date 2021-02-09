# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved. #
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from base.eval import AverageMeter, accuracy
from base.utils import isnan


def train_epoch(clser, train_loader, optimizer, epoch, use_gpu):
    """
    Training the model for one epoch.
    """
    clser.train()
    Cls_Criterion = torch.nn.CrossEntropyLoss()

    for step, batch_data in enumerate(train_loader):
        X = batch_data["data"].type(torch.FloatTensor)
        assert len(X.shape) == 2, X.shape
        GT = batch_data["label"].type(torch.long)
        if use_gpu:
            X = X.cuda()
            GT = GT.cuda()

        # Forward
        DTW_dist, logit_vec = clser(X)

        x_entropy = Cls_Criterion(logit_vec, GT)
        if isnan(x_entropy):
            raise ValueError("x_entropy is nan.")

        # Backward
        optimizer.zero_grad()
        x_entropy.backward()
        optimizer.step()


def val_epoch(clser, test_loader, use_gpu):
    """
    Eval the model error rate on test set.
    """
    clser.eval()
    top1 = AverageMeter("Acc@1", ":6.2f")

    for iter_idx, batch_data in enumerate(test_loader):
        X = batch_data["data"].type(torch.FloatTensor)
        assert len(X.shape) == 2, X.shape

        if use_gpu:
            X = X.cuda()

        DTW_dist, logit = clser(X)
        if use_gpu:
            logit = logit.cpu()
        label = batch_data["label"].type(torch.LongTensor)

        acc1 = accuracy(logit, label, topk=(1,))
        top1.update(acc1[0].item(), label.shape[0])

    return 1.0 - top1.avg / 100.0
