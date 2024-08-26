import pandas as pd 
import torch
from chronos import ChronosPipeline
import matplotlib.pyplot as plt  # requires: pip install matplotlib
import numpy as np




''' close price pre 400 different window plots '''
df = pd.read_csv("SPX2.csv")

acc_count = []

for window in [-80,-60,-40]:#[-30,-25,-15,-10,-5]:
    span = 400
    start_idx = len(df) + window
    forward = 1
    acc = 0

    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-base",
        device_map="cuda",  
        torch_dtype=torch.bfloat16,
    )

    medians = []
    lows = []
    highs = []

    for j in range(start_idx,len(df)+1):
        data = df["Close"][j-span:j]
        assert len(data) == span
        forecast = pipeline.predict(
            context=torch.tensor(data.to_numpy()),
            prediction_length=forward,
            num_samples=100,
        )
        low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
        medians.append(median)
        lows.append(low)
        highs.append(high)
        print(f'{j}/{len(df)} done')

    medians = [arr.tolist() for arr in medians]
    # lows = [arr.tolist() for arr in lows]
    # highs = [arr.tolist() for arr in highs]

    plt.figure(figsize=(8, 4))
    plt.plot(range(window,1), df["Close"][start_idx-1:], color="royalblue", label="ground truth")
    colors = ['tomato', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    for i in range(len(medians)):
        days = range(window+i, window+i+forward+1)
        value_to_add = df["Close"][start_idx - 1 + i]

        if i == 0:
            plt.plot(days, [value_to_add] + medians[i], color=colors[i % len(colors)], ls=':', label=f"median forecast for {forward}-day forward")
            # plt.fill_between(days, [value_to_add] + lows[i], [value_to_add] + highs[i], color=colors[i % len(colors)], alpha=0.3, label="80% prediction interval")
        else:
            plt.plot(days, [value_to_add] + medians[i], color=colors[i % len(colors)], ls=':')
            # plt.fill_between(days, [value_to_add] + lows[i], [value_to_add] + highs[i], color=colors[i % len(colors)], alpha=0.3)

        if i != len(medians)-1:
            if np.sign(medians[i] - value_to_add) == np.sign(df["Close"][start_idx + i] - value_to_add):
                acc += 1

    plt.legend()
    plt.grid()
    plt.savefig(f'pre{span}_window{abs(window)}_forward{forward}_smp100_hit{acc}.png') 

    # acc_count.append(acc)
    # print(f'{span}/1000 done')


# plt.figure(figsize=(8, 4))
# plt.plot(range(50,1000,20), np.array(acc_count), color="royalblue", label="right direction prediction count")
# plt.legend()
# plt.grid()
# plt.savefig(f'forward{forward}_smp100_hit_count.png') 

    