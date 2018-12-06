from stockInput import *

# Get the files' names containing our target stocks' market prices
targets = ['AAPL', 'MSFT', 'GOOGL', 'FB', 'AMZN', 'INTC', 'NVDA', 'AMD']
stocks = []
for i in range(0,len(targets)):
  stocks.append(list(Path('./data/stocks/Stocks').rglob(targets[i].lower()+'.us.[tT][xX][tT]'))[0])
stocks.append(list(Path('./data/stocks').rglob('*.[cC][sS][Vv]'))[0])
stocks.append(list(Path('./data/stock_exchange').rglob('prices.[cC][sS][Vv]'))[0])

for s in stocks:
  print(s)

  # Function to make showing timestamps in a readable format easier
def convert_to_ticks(x, n=6, date_format='%Y-%m-%d'):
  step = int(len(x)/n)
  ticks_ts = []
  ticks_string = []
  for i in range(n):
    ts = x[i * step]
    ticks_ts.append(ts)
    ticks_string.append(datetime.utcfromtimestamp(ts).strftime(date_format))
  ticks_ts.append(x[-1])
  ticks_string.append(datetime.utcfromtimestamp(x[-1]).strftime(date_format))
  return ticks_ts, ticks_string

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
plt.title("Stock pairs' correlation on their common timespan", fontsize=20)
ax = sns.heatmap(df, linewidths=.5, ax=ax, cmap='brg', vmin=-1.0, vmax=1.0)

# Drawing diagrams of the opening prices and the price changes of
# the stocks by companies
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
    ax[i].set_ylabel('Price')
    ax[i].set_title(key + title[i])

  plt.show()
