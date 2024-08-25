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
We utilize techniques from the ICDAR 2023 paper, `LineFormer - Rethinking Chart Data Extraction as Instance Segmentation`, to extract line data through instance segmentation. The code is adapted from [LineFormer](https://github.com/TheJaeLal/LineFormer), and we use their pre-trained model directly for this task.

### 2) The right look-back window

Given historical stock price data, a critical question is determining how many past prices to use for forecasting future prices. We conducted a test to predict the next-day price movement for the final 10, 20, 30, 40, and 50 instances of the Close price. In this test, we used look-back windows ranging from 10 to 1000. 

For example, when predicting the next-day price movement for the final 10 instances of the Close price, if we choose a window size of 300, the prediction for the next-day movement of each instance is based on its previous 300 prices.

After plotting the number of correct movement predictions against the size of the look-back window, we found that a look-back window size of 400 consistently led to the highest number of correct predictions across all instance sizes. An example of the prediction for the final 20 instances of the Close price is shown below, where when the window size is around 400, movement predictions for 15 out of 20 instances are correct.

![](./look_back_window/close_forward1_smp100_hit_countof20.png.png)

Other plots can be found at `./look_back_window`.

### 3) The right foreward window 


### 4) The results

## 2. LLM + LSTM

All results can be reproduce using the code `matrix_produce.py`
