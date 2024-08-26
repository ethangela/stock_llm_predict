import pandas as pd 
import torch
from chronos import ChronosPipeline
import matplotlib.pyplot as plt 
import numpy as np
import pickle

'''data'''
# df = pd.read_csv("SPX1.csv")
df = pd.read_csv("SPX2.csv")


'''open high low ground truth check'''
# plt.figure(figsize=(8, 4))
# plt.plot(df["Open"], color="tomato", label="ground-truth Open")
# plt.plot(df["High"], color="green", label="ground-truth High")
# plt.plot(df["Low"], color="orange", label="ground-truth Low")
# plt.plot(df["Close"], color="royalblue", label="ground-truth Close")
# plt.legend()
# plt.grid()
# plt.savefig('test_spx_gt.png')  

# df_shifted = df.shift(1)
# trend_open = (df['Open'] > df_shifted['Open']) == (df['Close'] > df_shifted['Close'])
# trend_high = (df['High'] > df_shifted['High']) == (df['Close'] > df_shifted['Close'])
# trend_low = (df['Low'] > df_shifted['Low']) == (df['Close'] > df_shifted['Close'])
# consistent_trend = trend_open & trend_high & trend_low

# print(trend_open[trend_open == False].index.tolist())
# print(trend_open[trend_high == False].index.tolist())
# print(trend_open[trend_low == False].index.tolist())



'''zero-shot one day forward prediction'''
# pipeline = ChronosPipeline.from_pretrained(
#     "amazon/chronos-t5-base",
#     device_map="cuda",  
#     torch_dtype=torch.bfloat16,
# )

# medians = []
# lows = []
# highs = []

# for i in range(101,len(df)+1):
#     data = df["Close"][:i]

#     forecast = pipeline.predict(
#         context=torch.tensor(data),
#         prediction_length=1,
#         num_samples=20,
#     )
#     low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
#     medians.append(median)
#     lows.append(low)
#     highs.append(high)
#     print(f'{i}/{len(df)} done')

# with open('medians.pkl', 'wb') as file:
#     pickle.dump(medians, file)
# with open('lows.pkl', 'wb') as file:
#     pickle.dump(lows, file)
# with open('highs.pkl', 'wb') as file:
#     pickle.dump(highs, file)

# with open('medians.pkl', 'rb') as file:
#     medians = pickle.load(file)
# with open('lows.pkl', 'rb') as file:
#     lows = pickle.load(file)
# with open('highs.pkl', 'rb') as file:
#     highs = pickle.load(file)

# medians = [arr.item() for arr in medians]
# lows = [arr.item() for arr in lows]
# highs = [arr.item() for arr in highs]
# forecast_index = range(101,len(df)+1)

# plt.figure(figsize=(8, 4))
# plt.plot(df["Close"], color="royalblue", label="ground truth")
# plt.plot(forecast_index, medians, color="tomato", label="median forecast")
# plt.fill_between(forecast_index, lows, highs, color="tomato", alpha=0.3, label="80% prediction interval")
# plt.legend()
# plt.grid()
# plt.savefig('test_spx_step.png') 


'''one day difference'''
# plt.figure(figsize=(8, 4))
# plt.plot(medians[:-1] - df["Close"][101:], color="royalblue", label="difference between gt and predict")
# plt.legend()
# plt.grid()
# plt.savefig('test_spx_diff.png') 



'''zero-shot multi-day forward prediction single plot '''
# for forward in range(1,2):
#     start_idx = 678 #678 3700 
#     dirs = '1pkls' #1pkls 2pkls
#     total = 700 # 700 3721
#     dir_num = 1 # 1 2
#     window = -23 #-23 -22

#     # pipeline = ChronosPipeline.from_pretrained(
#     #     "amazon/chronos-t5-base",
#     #     device_map="cuda",  
#     #     torch_dtype=torch.bfloat16,
#     # )

#     # medians = []
#     # lows = []
#     # highs = []

#     # for i in range(start_idx,len(df)+1):
#     #     data = df["Close"][:i]

#     #     forecast = pipeline.predict(
#     #         context=torch.tensor(data),
#     #         prediction_length=forward,
#     #         num_samples=20,
#     #     )
#     #     low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
#     #     medians.append(median)
#     #     lows.append(low)
#     #     highs.append(high)
#     #     print(f'{i}/{len(df)} done')

#     # with open(f'{dirs}/medians{forward}.pkl', 'wb') as file:
#     #     pickle.dump(medians, file)
#     # with open(f'{dirs}/lows{forward}.pkl', 'wb') as file:
#     #     pickle.dump(lows, file)
#     # with open(f'{dirs}/highs{forward}.pkl', 'wb') as file:
#     #     pickle.dump(highs, file)

#     with open(f'{dirs}/medians{forward}.pkl', 'rb') as file:
#         medians = pickle.load(file)
#     # with open(f'{dirs}/lows{forward}.pkl', 'rb') as file:
#     #     lows = pickle.load(file)
#     # with open(f'{dirs}/highs{forward}.pkl', 'rb') as file:
#     #     highs = pickle.load(file)

#     medians = [arr.tolist() for arr in medians]
#     # lows = [arr.tolist() for arr in lows]
#     # highs = [arr.tolist() for arr in highs]


#     plt.figure(figsize=(8, 4))
#     plt.plot(range(window,1), df["Close"][start_idx-1:], color="royalblue", label="ground truth")
#     colors = ['tomato', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
#     for i in range(len(medians)):
#         days = range(start_idx-1+i-total, start_idx+i+forward-total)
#         value_to_add = df["Close"][start_idx - 1 + i]
#         if i == 0:
#             plt.plot(days, [value_to_add] + medians[i], color=colors[i % len(colors)], ls=':', label=f"median forecast for {forward}-day forward")
#             # plt.fill_between(days, [value_to_add] + lows[i], [value_to_add] + highs[i], color=colors[i % len(colors)], alpha=0.3, label="80% prediction interval")
#         else:
#             plt.plot(days, [value_to_add] + medians[i], color=colors[i % len(colors)], ls=':')
#             # plt.fill_between(days, [value_to_add] + lows[i], [value_to_add] + highs[i], color=colors[i % len(colors)], alpha=0.3)


#     plt.legend()
#     plt.grid()
#     plt.savefig(f'{dir_num}test_spx_step{forward}_test.png') 
    




'''zero-shot multi-day forward prediction average plot '''
# medians_avg = []
# lows_avg = []
# highs_avg = []

# start_idx = 3700 #678 3700 
# dirs = '2pkls' #1pkls 2pkls
# total = 3721 # 700 3721
# dir_num = 2 # 1 2
# window = -22 #-23 -22

# for forward in range(1,3):
#     with open(f'{dirs}/medians{forward}.pkl', 'rb') as file:
#         medians = pickle.load(file)
#     with open(f'{dirs}/lows{forward}.pkl', 'rb') as file:
#         lows = pickle.load(file)
#     with open(f'{dirs}/highs{forward}.pkl', 'rb') as file:
#         highs = pickle.load(file)

#     medians_avg.append([arr.tolist()[0] for arr in medians])
#     # lows_avg.append([arr.tolist() for arr in lows])
#     # highs_avg.append([arr.tolist() for arr in highs])

# medians = np.average(np.array(medians_avg), axis=0)
# medians = [[item] for item in medians]

# plt.figure(figsize=(8, 4))
# plt.plot(range(window,1), df["Close"][start_idx-1:], color="royalblue", label="ground truth")
# colors = ['tomato', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
# for i in range(len(medians)):
#     days = range(start_idx-1+i-total, start_idx+i+1-total)
#     value_to_add = df["Close"][start_idx - 1 + i]
#     if i == 0:
#         plt.plot(days, [value_to_add] + medians[i], color=colors[i % len(colors)], ls=':', label=f"median forecast for 1-day forward")
#         # plt.fill_between(days, [value_to_add] + lows[i], [value_to_add] + highs[i], color=colors[i % len(colors)], alpha=0.3, label="80% prediction interval")
#     else:
#         plt.plot(days, [value_to_add] + medians[i], color=colors[i % len(colors)], ls=':')
#         # plt.fill_between(days, [value_to_add] + lows[i], [value_to_add] + highs[i], color=colors[i % len(colors)], alpha=0.3)

# plt.legend()
# plt.grid()
# plt.savefig(f'{dir_num}test_spx_avg.png') 

