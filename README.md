# Assignment 3

## 0. Introduction

This assignment focuses on leveraging LLMs for SPX index's Close price time series forecasting. We introduce everythin in two sections:

1. **Zero-shot LLM: We directly try forecasting with LLM in a zero-shot fashion**
   - The LLM is applied directly to predict unseen SPX data without prior training on such stock prices.
   - We input the sequence of historical Close prices and output the forecasted Close price.

2. **LLM + LSTM: We use LLM-generated sequence for feature engineering and build a model for forecasting**
   - First, generate forecast sequences for Open, High, and Low (OHL) prices using the LLM.
   - Then, build and select the top 100 features from historical OHL prices, and similarly build the corresponding top 100 features for forecasted OHL prices.
   - Finally, these features are used to train an LSTM model for predicting the directional movement of the Close price.

## 1. Zero-shot LLM

### 1) The right model 

After reviewing various SOTA LLMs, we found the latest Chronos, a pretrained time series forecasting model, generates the most remarkable zero-shot performance [Performance](https://github.com/amazon-science/chronos-forecasting). We therefore use Chronos as the backbone for all tasks below, where for each prediction we generate 100 samples and use the median as the output. 

### 2) The right rolling look-back window

Determining the optimal number of past prices for forecasting is crucial. We tested rolling look-back windows from 10 to 1000 to predict the next-day price movement for the final 10, 20, 30, 40, and 50 Close price instances. For example, using a 300-day window means each prediction is based on the previous 300 prices. 

Our analysis revealed that a 400-day look-back window consistently yielded the highest accuracy across all instance sizes. The plot below illustrates this with the final 20 instances, where correct predictions peak at a window size of 400.

![](./look_back_window/close_forward1_smp100_hit_countof20.png)

Other plots can be found at `./look_back_window`. In the following tasks, we stick to window size 400.

### 3) The right foreward window 


### 4) The results

## 2. LLM + LSTM

All results can be reproduce using the code `matrix_produce.py`
