# Assignment 3

## Introduction

This assignment focuses on leveraging LLMs for SPX index's Close price time series forecasting.

1. **Zero-shot LLM: We directly try forecasting with LLM in a zero-shot fashion**
   - The LLM is applied directly to predict unseen SPX data without prior training on such stock prices.
   - We input the sequence of historical Close prices and output the forecasted Close price.

2. **LLM + LSTM: We use LLM-generated sequence for feature engineering and build a model for forecasting**
   - First, generate forecast sequences for Open, High, and Low (OHL) prices using the LLM.
   - Then, build and select the top 100 features from historical OHL prices, and similarly build the corresponding top 100 features for forecasted OHL prices.
   - Finally, these features are used to train an LSTM model for predicting the directional movement of the Close price.

## 1. Zero-shot LLM

### 1) The right model 
We utilize techniques from the ICDAR 2023 paper, `LineFormer - Rethinking Chart Data Extraction as Instance Segmentation`, to extract line data through instance segmentation. The code is adapted from [LineFormer](https://github.com/TheJaeLal/LineFormer), and we use their pre-trained model directly for this task.

### 2) The right backward window

![Project Screenshot](./images/screenshot.png)
Coordinates and replots of all 6 tickers can be found at `./chart_extract_raw_output`.

### 3) The right foreward window 

The extracted coordinates of the k-line, EMA, and WMA lines for each ticker may vary in length, which we need to address before calculating similarity. For example, when calculating the Pearson correlation between two lines, they must have the same length. The details of the lengths of the k-line, EMA, and WMA lines for each ticker can be found in `./length.txt`.

First, we observed significant fluctuations in the lengths of the EMA lines, so we excluded EMA lines from the similarity analysis. Next, we found that, except for `k_line_1` (the extracted k-line for ticker 1), all other k-lines have a similar number of coordinates. All wma-lines also have a similar number of coordinates. We manually identified that the existence of some sparse points in `k_line_1` results such lower number of coordinates. Consdiering this, whenever dealing with `k_line_1`, we use interpolation to increase its length. An example of this interpolation is shown below:

### 4) The results

## 2. LLM + LSTM

All results can be reproduce using the code `matrix_produce.py`
