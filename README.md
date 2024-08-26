# Assignment 3



## 0. Introduction

This assignment focuses on leveraging LLMs for SPX index's `Close` price time series forecasting. We introduce everything in two sections:

1. **Zero-shot LLM: We directly try forecasting with LLM in a zero-shot fashion**
   - The LLM is applied directly to predict unseen SPX data without prior training on such stock prices.
   - We input the sequence of historical `Close` prices and output the forecasted `Close` price.

2. **LLM + LSTM: Feature Engineering with LLM and Forecasting with LSTM**
   - First, generate and select the top 100 features derived from historical `OHLC` prices.
   - Then, using historical data, train an LSTM model with these features to predict the directional movement of the historical `Close` price.
   - Finally: utilize the LLM to forecast the top 100 features for the upcoming days and apply the trained LSTM model to predict the directional movement of the `Close` price for those future days.



## 1. Zero-shot LLM (`chronos_various_test.py` and `chronos_main.py`)

#### 1) The right model 

After reviewing various SOTA LLMs, we found the latest [Chronos](https://github.com/amazon-science/chronos-forecasting), a pretrained time series forecasting model, generates the most remarkable zero-shot performance. We therefore use Chronos as the backbone for all tasks below, where for each prediction we generate 100 samples and use the median as the output. 


#### 2) The right rolling look-back window

Determining the optimal number of past prices for future price forecasting is crucial. We tested the size of rolling look-back windows from 10 to 1000, e.g., using a 300-day window means each prediction is based on the previous 300 prices. 

We test the prediction of the next-day `Close` price for the final 10, 20, 30, 40, and 50 days in the given `SPX.csv`.

Our analysis revealed that a 400-day look-back window consistently yielded the highest accuracy across all test days. The plot below illustrates this with the final 20 days, where correct predictions peak (15) at a window size of 400.

![](./look_back_window/close_forward1_smp100_hit_countof20.png)

Other plots can be found at `./look_back_window`. In the following tasks, we stick to look-back window with size 400.


#### 3) The right forward window 

Determining the optimal number of future prices to forecast based on a fixed number of past prices is equally important. Our analysis showed that predicting just one day ahead provides the most proper results, while longer forecasts tend to deviate significantly from the actual values.

Below are the comparable examples of one-day (top) and two-day (bottom) forward predictions:

![](./look_forward_window/2test_spx_step1_test.png)

![](./look_forward_window/2test_spx_step2_test.png)

Other forward windows can be found at `./look_forward_window`.

Nevertheless, we can still do multi-day forecasts using a step-by-step approach: first, predict the next day, then use that prediction as the basis for the following day, rolling the look-back window forward for each subsequent day.


#### 4) The results for next four days

Using a fixed look-back window of `400` and a forward window of `1`, we input the `Close` price movement into the Chronos LLM and obtained predictions for the next four days.

The actual `Close` prices on the last given day along with the predicted prices for the next four days are 4769.83, 4772.49, 4774.26, 4776.10, and 4778.05, resulting in the next four-day `Close` price movements as: `up`, `up`, `up`, and `up`.




## 2. LLM + LSTM (`feature.py` and `predict.py`)

#### 1) Features from `OHLC`

Inspired by a 2021 PhD [thesis](https://discovery.ucl.ac.uk/id/eprint/10155501/2/AndrewDMannPhDFinal.pdf), we mine features from `OHLC` using all possible combinations of differences, ratios, and pairwise operations of daily `OHLC` data with L lags. Specifically:
   - Lags (L): Refers to the number of days considered, e.g., L=2 includes today (0) and yesterday (1).
   - Differences: Calculations of differences between `OHLC` values at various lags, e.g., `Close0 - Low1`.
   - Ratios: Ratios of `OHLC` values at different lags, e.g., `Low0 / Low1`.
   - Pairwise Operations: Operations (both differences and ratios) between features derived from Differences and Ratios, e.g., `(Close0 - Low1) / (Low0 / Low1)`.


#### 2) Rank and select the top 100 features
We set L=2 and generated over 800 features. XGBoost is then employed to rank each feature's importance with respect to predicting the next-day `Close` price directional change, using the total gain metric.

The importance curve is illustrated below:

![](./lstm/importance.png)

We selected the top 100 features, and their names are listed in `./lstm/top100.txt.`


#### 3) Build LSTM model

With 100 features selected, we built an LSTM model to predict the directional movement of the `Close` price using historical data. Due to the limited dataset size (3720 days), we opted for a simple network structure with few layers.

We split the historical data into training and testing sets. After 20 epochs of training, the model achieved an accuracy of 66.15% on the test set. We anticipate that accuracy could improve with additional data in the future (e.g., hourly `OHLC`) and more complex models (e.g., Transformers).


#### 4) Use LLM to generate next-day features

We now have sequences of 100 features for all historical data. Each sequence can be input into the LLM to predict values for the next few days, using the same LLM backbone, look-back window, and forward window as described previously.

An example of the performance of LLM in predicting next-day feature ((Open0-Low1)/(High0-Close1)) is shown below:

![](./feature/pre400_window20_forward1_smp100_hit11_feature_rank_1.png)

#### 5) The results for next four days

Finally, using features generated from the LLM and the trained LSTM model, we predict the next four days of `Close` price movements as: `down`, `down`, `up`, and `up`.
