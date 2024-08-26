import pandas as pd
import numpy as np
from itertools import combinations, product
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from chronos import ChronosPipeline
import pickle




'''Load data'''
df = pd.read_csv("SPX2.csv")



'''feature selection'''
# Add lags for 0 (today) and 1 (yesterday)
df['Open0'] = df['Open']
df['High0'] = df['High']
df['Low0'] = df['Low']
df['Close0'] = df['Close']

for lag in range(1, 2):  
    df[f'Open{lag}'] = df['Open'].shift(lag)
    df[f'High{lag}'] = df['High'].shift(lag)
    df[f'Low{lag}'] = df['Low'].shift(lag)
    df[f'Close{lag}'] = df['Close'].shift(lag)


# Generate Differences and Ratios with 2 lags
diff_features = []
ratio_features = []
cols = ['Open', 'High', 'Low', 'Close']
lags = ['0', '1']

for (col1, lag1), (col2, lag2) in combinations(product(cols, lags), 2):
    diff_name = f'({col1}{lag1}-{col2}{lag2})'
    diff_features.append((df[f'{col1}{lag1}'] - df[f'{col2}{lag2}'], diff_name))
    
    ratio_name = f'({col1}{lag1}/{col2}{lag2})'
    ratio_features.append((df[f'{col1}{lag1}'] / df[f'{col2}{lag2}'], ratio_name))

diff_df = pd.concat([f[0].rename(f[1]) for f in diff_features], axis=1)
ratio_df = pd.concat([f[0].rename(f[1]) for f in ratio_features], axis=1)


# Generate Pairwise Operations between Differences and Ratios
pairwise_features = []

for (f1, name1), (f2, name2) in combinations(diff_features + ratio_features, 2):
    pairwise_features.append((f1 - f2, f'({name1}-{name2})'))
    
    pairwise_features.append((f1 / f2, f'({name1}/{name2})'))


# Convert pairwise_features to DataFrame
pairwise_df = pd.concat([f[0].rename(f[1]) for f in pairwise_features], axis=1)
features_df = pd.concat([diff_df, ratio_df, pairwise_df], axis=1)


# Create the Target (Close price directional change)
df['predict'] = df['Close'].shift(-1)
df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
# df = df.dropna()  # Drop rows with NaN values due to shift


# Train the Model
X = features_df.iloc[1:-1, :]  # All features except the last row
y = df['Target'].iloc[1:-1]  # Corresponding target variable
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.mean(), inplace=True)

# Split the data
split_index = int(0.8 * len(X)) 
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

#normlize 
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and fit the model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train_scaled, y_train)

# Predictions and accuracy
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Feature importance based on gain
original_columns = features_df.columns
new_feature_names = [f"f{i}" for i in range(len(original_columns))]
column_mapping = dict(zip(new_feature_names, original_columns))

importance = model.get_booster().get_score(importance_type='total_gain')
importance_df = pd.DataFrame(importance.items(), columns=['Feature', 'Total Gain']).sort_values(by='Total Gain', ascending=False)
importance_df['Feature'] = importance_df['Feature'].map(column_mapping)

# Show the most important features
top100_features = importance_df.head(100)

# plot the distribution of importances 
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1,len(importance_df)+1), importance_df['Total Gain'], marker='o')  # Rank starts at 1
plt.title('Total Gain vs. Feature Rank')
plt.xlabel('Feature Rank')
plt.ylabel('Total Gain')
plt.grid(True)
plt.savefig(f'importance.png') 





'''feature flitering (depreciated)'''
# corr_ft = {}
# for feature in top100_features['Feature']:
#     corr_ft[feature] = X_train[feature].corr(y_train)

# corr_ft_df = pd.DataFrame(list(corr_ft.items()), columns=['Feature', 'corr_ft'])

# corr_ff = X_train[top100_features['Feature']].corr()

# def filter_features(importance_df, corr_ft_df, corr_ff, c1, c2):
#     filtered_ft = corr_ft_df[np.abs(corr_ft_df['corr_ft']) >= c1]
    
#     selected_features = []
    
#     for feature in filtered_ft['Feature']:
#         if all(np.abs(corr_ff[feature][selected_features]) <= c2 for selected_feature in selected_features):
#             selected_features.append(feature)
    
#     return selected_features

# c1 = 0.0
# c2 = 0.7  
# selected_features = filter_features(importance_df, corr_ft_df, corr_ff, c1, c2)
# print("Selected Features based on the thresholds c1 and c2:")
# print(selected_features)

# corr_values = corr_ff.values.flatten()
# mask = ~np.eye(corr_ff.shape[0], dtype=bool)
# corr_values_no_diag = corr_ff.values[mask]
# max_value = np.max(corr_values_no_diag)
# min_value = np.min(corr_values_no_diag)
# print(f"Maximum Correlation: {max_value:.4f}")
# print(f"Minimum Correlation: {min_value:.4f}")





'''100 features sequences next-day value produce'''
pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-base",
        device_map="cuda",  
        torch_dtype=torch.bfloat16,
    )



#next day predict
top_100_features = features_df[top100_features['Feature']].iloc[1:,:].reset_index(drop=True)
top_100_features.replace([np.inf, -np.inf], np.nan, inplace=True)
top_100_features.fillna(top_100_features.mean(), inplace=True)
print(top_100_features)

next_day_row = []
for idx, feature in enumerate(top_100_features.columns):

    span = 400
    forward = 1
    data = top_100_features[feature][-span:]
    assert len(data) == span
    forecast = pipeline.predict(
        context=torch.tensor(data.to_numpy()),
        prediction_length=forward,
        num_samples=100,
    )
    low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
    predict = median[0]
    next_day_row.append(predict)
    print(f'{idx} done.')

top_100_features = top_100_features.append(pd.Series(next_day_row, index=top_100_features.columns), ignore_index=True)
top_100_features.to_csv('top_100_features_forward2.csv', index=False) 



#next 2day predict
top_100_features = pd.read_csv("top_100_features_forward1.csv")
print(top_100_features)

next_day_row = []
for idx, feature in enumerate(top_100_features.columns):

    span = 400
    forward = 1
    data = top_100_features[feature][-span:]
    assert len(data) == span
    forecast = pipeline.predict(
        context=torch.tensor(data.to_numpy()),
        prediction_length=forward,
        num_samples=100,
    )
    low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
    predict = median[0]
    next_day_row.append(predict)
    print(f'{idx} done.')

top_100_features = top_100_features.append(pd.Series(next_day_row, index=top_100_features.columns), ignore_index=True)
print(top_100_features)
top_100_features.to_csv('top_100_features_forward2.csv', index=False) 



#next 3day predict
top_100_features = pd.read_csv("top_100_features_forward2.csv")
print(top_100_features)

next_day_row = []
for idx, feature in enumerate(top_100_features.columns):

    span = 400
    forward = 1
    data = top_100_features[feature][-span:]
    assert len(data) == span
    forecast = pipeline.predict(
        context=torch.tensor(data.to_numpy()),
        prediction_length=forward,
        num_samples=100,
    )
    low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
    predict = median[0]
    next_day_row.append(predict)
    print(f'{idx} done.')

top_100_features = top_100_features.append(pd.Series(next_day_row, index=top_100_features.columns), ignore_index=True)
print(top_100_features)
top_100_features.to_csv('top_100_features_forward3.csv', index=False) 
    



'''open/high/low/close sequence next-days value produce'''
pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-base",
        device_map="cuda",  
        torch_dtype=torch.bfloat16,
    )

for price_type in ['Open', 'High', 'Low', 'Close']:

    close_df = df[price_type].iloc[1:].reset_index(drop=True)
    print(close_df)

    span = 400
    forward = 1

    for _ in range(4):
        data = close_df[-span:]
        assert len(data) == span
        forecast = pipeline.predict(
            context=torch.tensor(data.to_numpy()),
            prediction_length=forward,
            num_samples=100,
        )
        low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
        predict = median[0]
        close_df = close_df.append(pd.Series(predict), ignore_index=True)

    print(close_df)
    close_df.to_csv(f'{price_type}_df_forward4.csv', index=False) 



'''close sequence one-step-next-days value produce'''
span = 400
forward = 4
data = close_df[-span:]
assert len(data) == span
forecast = pipeline.predict(
    context=torch.tensor(data.to_numpy()),
    prediction_length=forward,
    num_samples=100,
)
low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
predict = median
close_df = close_df.append(pd.Series(predict), ignore_index=True)
print(close_df)
close_df.to_csv('close_df_forward4_direct.csv', index=False) 




'''100 features sequences next-day value plot'''
for idx, feature_name in enumerate(top100_features['Feature']):

    feature_importance = idx+1
    df = features_df[feature_name].iloc[1:].reset_index(drop=True)

    acc_count = []
    window = -20
    span = 400
    start_idx = len(df) + window
    forward = 1
    acc = 0

    medians = []
    lows = []
    highs = []

    for j in range(start_idx,len(df)+1):
        data = df[j-span:j]
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
        # print(f'{j}/{len(df)} done')

    medians = [arr.tolist() for arr in medians]
    # lows = [arr.tolist() for arr in lows]
    # highs = [arr.tolist() for arr in highs]

    plt.figure(figsize=(8, 4))
    plt.plot(range(window,1), df[start_idx-1:], color="royalblue", label="ground truth")
    colors = ['tomato', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    for i in range(len(medians)):
        days = range(window+i, window+i+forward+1)
        value_to_add = df[start_idx - 1 + i]

        if i == 0:
            plt.plot(days, [value_to_add] + medians[i], color=colors[i % len(colors)], ls=':', label=f"median forecast for {forward}-day forward")
            # plt.fill_between(days, [value_to_add] + lows[i], [value_to_add] + highs[i], color=colors[i % len(colors)], alpha=0.3, label="80% prediction interval")
        else:
            plt.plot(days, [value_to_add] + medians[i], color=colors[i % len(colors)], ls=':')
            # plt.fill_between(days, [value_to_add] + lows[i], [value_to_add] + highs[i], color=colors[i % len(colors)], alpha=0.3)

        if i != len(medians)-1:
            if np.sign(medians[i] - value_to_add) == np.sign(df[start_idx + i] - value_to_add):
                acc += 1

    plt.legend()
    plt.grid()
    plt.title(f'{feature_name}')
    plt.savefig(f'./features_forcast_plots/pre{span}_window{abs(window)}_forward{forward}_smp100_hit{acc}_feature_rank_{feature_importance}.png') 

    print(f'{feature_importance}/100 done.')




