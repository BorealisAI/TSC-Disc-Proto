# TSC-Disc-Proto

This repo contains the official code of the project "Discriminative Prototypes learned by Dynamic Time Warping (DTW) for Time Series Classification (TSC)".

## 1. Dependent Packages and Platform

First we recommend to create a conda environment with all the required packages as follows,

```bash
conda env create -f environment.yml
conda activate TSC_Disc_Proto
# Then install Pytorch according to your infrastructure.
```
[Pytorch Install](https://pytorch.org/get-started/locally/) (tested with Ver. 1.7.1).

The configuration of these files was tested on Linux Platform with a GPU (RTX1080).



## 2. Download of Fish Dataset

In the paper, we train and evaluate our model on the Fish dataset. You can use the following command to easily download
the dataset:

```bash
bash get_dataset.sh
```

## 3. Run

Activating the conda environment `TSC_Disc_Proto` if not:

```bash
conda activate TSC_Disc_Proto
```

You can then use the following command to load the pretrained model and evaluate it on Fish dataset

```bash
python main.py
```

You should see the following lines in the terminal.

```text
#1. Build data loaders.
#2. Build the classifier with prototype sequences.
=====> Model on GPU(s).
7 of 7 parameters are trainable.
7 Classes 7 Weights to be trained in total.
#3. Build optimizer.
#4. Load the pretrained model.
#5. Evaluate
Test Error Rate: 7.43%.
```

Yoiu can use the following command to train a new model and evaluate it on Fish dataset:

```bash
python main.py --train
```

You should see the following lines in the terminal.

```text
#1. Build data loaders.
#2. Build the classifier with prototype sequences.
=====> Model on GPU(s).
7 of 7 parameters are trainable.
7 Classes 7 Weights to be trained in total.
#3. Build optimizer.
#4. Training
It took ~4 minutes to train.
#5. Evaluate
Test Error Rate: 7.43%.
```

Note that the test error rate may varied around 6.00% ~ 9.00%.

## 4. Result Comparison

|               | Error Rate (%) |
|:--------------|:--------------:|
| ED            | 21.71          |
| DTW           | 17.71          |
| DTW (opt)     | 15.43          |
| Ours (init)   | 33.71          |
| Ours          | **7.43**       |

The baseline DTW results are reported [Here](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/), ID 28 \(Fish
Dataset\) in the Table.

## 5. Citations and Links

UCR Datasets [Paper](https://link.springer.com/article/10.1007/s10618-016-0483-9):

```text
@article{bagnall16bakeoff,
title={The Great Time Series Classification Bake Off: a Review and Experimental Evaluation of Recent Algorithmic Advances},
author={A. Bagnall and J. Lines and A. Bostrom and J. Large and E. Keogh},
journal={Data Mining and Knowledge Discovery},
volume={31},
issue={3},
pages={606-660},
year={2017}
}
```

Links:

- [The UEA & UCR Time Series Classification Repository](http://www.timeseriesclassification.com)
- [Fish Dataset Details](http://www.timeseriesclassification.com/description.php?Dataset=Fish)
