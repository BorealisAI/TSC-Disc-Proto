#!/usr/bin/env bash

# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved. #
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

mkdir -p ./data
wget http://www.timeseriesclassification.com/Downloads/Fish.zip -P ./data/
unzip ./data/Fish.zip -d ./data/Fish/
rm -f ./data/Fish.zip
