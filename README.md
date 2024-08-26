# Assignment 3

## 0. Introduction

This assignment focuses on leveraging LLMs for SPX index's `Close` price time series forecasting. We introduce everythin in two sections:

1. **Zero-shot LLM: We directly try forecasting with LLM in a zero-shot fashion**
   - The LLM is applied directly to predict unseen SPX data without prior training on such stock prices.
   - We input the sequence of historical `Close` prices and output the forecasted `Close` price.

2. **LLM + LSTM: Feature Engineering with LLM and Forecasting with LSTM**
   - First, generate and select the top 100 features from historical `OHLC` prices.
   - Then, train an LSTM model on historical data using these features to predict the directional movement of the `Close` price.
   - Finally: utilize the LLM to forecast the top 100 features for the upcoming days and apply the trained LSTM model to predict the directional movement of the `Close` price for those days.

## 1. Zero-shot LLM

### 1) The right model 

After reviewing various SOTA LLMs, we found the latest [Chronos](https://github.com/amazon-science/chronos-forecasting), a pretrained time series forecasting model, generates the most remarkable zero-shot performance. We therefore use Chronos as the backbone for all tasks below, where for each prediction we generate 100 samples and use the median as the output. 

### 2) The right rolling look-back window

Determining the optimal number of past prices for forecasting is crucial. We tested rolling look-back windows from 10 to 1000 to predict the next-day price movement for the final 10, 20, 30, 40, and 50 Close price instances. For example, using a 300-day window means each prediction is based on the previous 300 prices. 

Our analysis revealed that a 400-day look-back window consistently yielded the highest accuracy across all instance sizes. The plot below illustrates this with the final 20 instances, where correct predictions peak (15) at a window size of 400.

![](./look_back_window/close_forward1_smp100_hit_countof20.png)

Other plots can be found at `./look_back_window`. In the following tasks, we stick to window size 400.

### 3) The right forward window 

Determining the optimal number of future prices to forecast based on a fixed number of past prices is equally important. Our analysis showed that predicting just one day ahead provides the most accurate results, while longer forecasts tend to deviate significantly from the actual values.

For multi-day forecasts, we recommend a step-by-step approach: first, predict the next day, then use that prediction as the basis for the following day, rolling the look-back window forward for each subsequent day.

Below are examples of one-day (top) and two-day (bottom) forward predictions:

![](./look_forward_window/2test_spx_step1_test.png)

![](./look_forward_window/2test_spx_step2_test.png)

Other forward windows can be found at `./look_forward_window`.

### 4) The results for next four days

Using a fixed look-back window of `400` and a forward window of `1`, we input the `Close` price movement into the Chronos LLM and obtained predictions for the next four days.

The actual `Close` prices along with the predicted prices for the next four days are 4769.83, 4742.83, 4704.81, 4688.68, and 4697.24, resulting in the following next four-day `Close` price movements: `down`, `down`, `down`, and `up`.


## 2. LLM + LSTM

### 1) Features from `OHLC`

Inspired by a 2021 PhD [thesis](https://discovery.ucl.ac.uk/id/eprint/10155501/2/AndrewDMannPhDFinal.pdf), we mine features from `OHLC` using all possible combinations of differences, ratios, and pairwise operations of daily `OHLC` data with L lags. Specifically:
   - Lags (L): Refers to the number of days considered, e.g., L=2 includes today and yesterday.
   - Differences: Calculations of differences between `OHLC` values at various lags, e.g., `close0 - low1`.
   - Ratios: Ratios of `OHLC` values at different lags, e.g., `low0 / low1`.
   - Pairwise Operations: Operations (both differences and ratios) between features derived from Differences and Ratios, e.g., `(close0 - low1) / (low0 / low1)`.

### 2) Rank and select the top 100 features
We set L=2 and generated over 800 features. XGBoost was employed to rank each feature's importance with respect to predicting the next-day `Close` price directional change, using the total gain metric.

The importance curve is illustrated below:

![](./lstm/importance.png)

We selected the top 100 features, and their names are listed in `./lstm/top100.txt.`

### 3) Build LSTM model

With 100 features selected, we build a LSTM model to predict the directional movement of the `Close` price on historical data. Given the limited size of historical data (3720 days in total), we avoid complex network structure or multi layers, and the detials of network structure can be found in `predict.py`.

We break the historcial data into train and test set, after 20 epoches of training on train data, the predcition accuracy on test set is 66.15%. We note that this accuracy has potential to improve with more data in the future and more complexed models (i.e., Transformer). 

With 100 features selected, we built an LSTM model to predict the directional movement of the `Close` price using historical data. Due to the limited dataset size (3720 days), we opted for a simple network structure with few layers. Details of the network architecture are provided in `predict.py`.

We split the historical data into training and testing sets. After 20 epochs of training, the model achieved an accuracy of 66.15% on the test set. We anticipate that accuracy could improve with additional data in the future and more complex models, such as Transformers.

### 4) Use LLM to generate next-day features



### 4) The results for next four days
