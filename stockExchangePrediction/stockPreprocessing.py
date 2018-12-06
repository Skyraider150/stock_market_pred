from stockVisualisation import *

# Scalers to transform values into [0, 1] interval and inverse transform back to normal values
scalers = {}

# Normalizing datasets
for k, df in stock_dict.items():
  scaler = MinMaxScaler()
  df[['Open', 'High', 'Low', 'Close', 'Volume']] = \
    scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])
  scalers[k] = scaler

  # Selecting timespans where we have data from all companies
start_at = []
for k, df in stock_dict.items():
  start_at.append(df.values[0, 0])

for k, df in stock_dict.items():
  df = df.loc[df['Timestamp'] >= max(start_at)]
  df = df.reset_index(drop=True)

# Visualize common timespan (scaled values)
title = [' opening price', ' daily change']

for key in stock_dict.keys():
  x = stock_dict[key]['Timestamp']
  y = [
      stock_dict[key]['Open'],
      stock_dict[key]['Open'] - stock_dict[key]['Close']
  ]

  timestamp, date_string = convert_to_ticks(x.values, n=6)

  fig, ax = plt.subplots(1, 2, figsize=(22, 7))
  for i in range(2):
    ax[i].plot(x, y[i])
    ax[i].set_xticks(timestamp)
    ax[i].set_xticklabels(date_string)
    ax[i].set_xlabel('Date')
    ax[i].set_title(key + title[i])

  plt.show()

# +++
for key in stock_dict.keys():
  a = stock_dict[key].values
  b = (stock_dict[key]['Open']-stock_dict[key]['Close']).values
  b = np.reshape(b, (-1, 1))
  sc = StandardScaler()
  sc.fit(b)
  b = sc.transform(b)
  #b = b * [10]*b.shape[0]
  c = np.concatenate((a, b), 1)
  c = pd.DataFrame(c, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Diff'])
  stock_dict[key] = c

plt.plot(stock_dict['GOOGL']['Diff'])

# Calculate Pearson correlation coefficients
stock_correlations = {}
list_of_names = list(stock_dict.keys())

for i1 in range(len(list_of_names)):
  for i2 in range(i1+1, len(list_of_names)):
    # We iterate through the companies in a way that all pairings only happen once
    key1 = list_of_names[i1]
    key2 = list_of_names[i2]
    dataframe1 = stock_dict[key1]
    dataframe2 = stock_dict[key2]
    # We calculate the time frame intersection between the data of the two companies
    ins = set(dataframe1.values[:, 0]).intersection(set(dataframe2.values[:, 0]))
    df1_kozos = dataframe1.loc[dataframe1['Timestamp'].isin(ins)]
    df2_kozos = dataframe2.loc[dataframe2['Timestamp'].isin(ins)]
    # We calculate and store the calculated results
    tmp = pearsonr(df1_kozos.values[:, 1], df2_kozos.values[:, 1])
    stock_correlations[(key1, key2)] = [tmp[0], tmp[1]]

# Preparing correlation data for seaborn heatmap
stock_table = np.full(shape=(8, 8), fill_value=np.NaN)
for i1 in range(len(list_of_names)):
  for i2 in range(i1+1, len(list_of_names)):
    key1 = list_of_names[i1]
    key2 = list_of_names[i2]
    stock_table[i1, i2] = stock_correlations[(key1, key2)][0]
# Create pandas dataframe from the correlation 2D array
df = pd.DataFrame(stock_table, index=list_of_names, columns=list_of_names)
# Seaborn heatmap requires the axes to be named
df = df.rename_axis('Stocks A')
df = df.rename_axis('Stocks B', axis="columns")
# First columns contains only NaN, so we can drop it
df = df.drop(df.columns[0], axis=1)
# Last row contains only NaN, so we can drop it
df = df.drop(df.index[len(df)-1])

# Drawing a Seaborn heatmap for the correlations
f, ax = plt.subplots(figsize=(10, 10))
plt.title("Stock pairs' correlation\n2012.05.18. - 2017.11.10.", fontsize=24)
ax = sns.heatmap(df, linewidths=.5, ax=ax, cmap='brg', vmin=-1.0, vmax=1.0)

# Train-Valid_Test split
valid_split = 0.7
test_split = 0.85

train_stock, valid_stock, test_stock = {}, {}, {}

for k, df in stock_dict.items():
  vals = df.values
  vs = int(len(vals) * valid_split)
  ts = int(len(vals) * test_split)
  train_stock[k] = vals[:vs]
  valid_stock[k] = vals[vs:ts]
  test_stock[k] = vals[ts:]
  print(k)
  print('\ttrain length:', len(train_stock[k]))
  print('\tvalid length:', len(valid_stock[k]))
  print('\ttest  length:', len(test_stock[k]))
