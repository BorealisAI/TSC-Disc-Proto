# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved. #
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.optim as optim

from base.datasets import build_loader
from base.networks import proto_clser
from base.run import train_epoch, val_epoch



def main(args):
    data_rt = "./data"
    dataset_rt = "{}/Fish".format(data_rt)
    train_data_file = "{}/Fish_TRAIN.txt".format(dataset_rt)
    test_data_file = "{}/Fish_TEST.txt".format(dataset_rt)
    t_ckpt_file = "{}/t_m_ckpt.pth".format(data_rt)
    pt_ckpt_file = "{}/pt_m_ckpt.pth".format(data_rt)
    Class_N = 7
    train_bz = 35
    test_bz = 10
    DTW_w_p = 0.1
    lr = 0.05
    opt_type = "ADAM"
    lr_decay = 0.2
    lr_decay_epoch = 25
    total_epoch = 60
    use_gpu = True
    TRAIN_FLAG = args.train

    print("#1. Build data loaders.")
    train_loader, Train_N = build_loader(train_data_file, Class_N, train_bz, shuffle=False)
    test_loader, Test_N = build_loader(test_data_file, Class_N, test_bz, shuffle=False)

    print("#2. Build the classifier with prototype sequences.")
    builder = proto_clser(train_loader, Train_N, Class_N, DTW_w_p)
    clser = builder.build()
    if use_gpu:
        clser = clser.cuda()
        print("=====> Model on GPU(s).")

    print("#3. Build optimizer.")
    if opt_type == "ADAM":
        optimizer = optim.Adam(clser.parameters(), lr=lr, eps=1e-5)
    else:
        raise ValueError("Undefined Optimizer: {}".format(opt_type))
    lr_scher = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_epoch, gamma=lr_decay)

    if TRAIN_FLAG:
        print("#4. Training")
        start_t = time.time()
        for epoch in range(total_epoch):
            train_epoch(clser, train_loader, optimizer, epoch, use_gpu)
            lr_scher.step()
        end_t = time.time()
        elapsed_t = end_t - start_t
        print("It took ~{} minutes to train.".format(int(elapsed_t / 60.0)))
        torch.save(clser.state_dict(), t_ckpt_file)
    else:
        print("#4. Load the pretrained model.")
        clser.load_state_dict(torch.load(pt_ckpt_file))

    print("#5. Evaluate")
    with torch.no_grad():
        val_err = val_epoch(clser, test_loader, use_gpu)
    print("Test Error Rate: {:.2f}%.".format(val_err * 100.0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Time Series Classification.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--train", default=False, action="store_true")
    args = parser.parse_args()
    main(args)
