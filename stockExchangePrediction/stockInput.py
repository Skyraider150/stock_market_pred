# Get the files' names containing our target stocks' market prices
from setup import *
from pathlib import Path

targets = ['AAPL', 'MSFT', 'GOOGL', 'FB', 'AMZN', 'INTC', 'NVDA', 'AMD']
stocks = []
for i in range(0, len(targets)):
    stocks.append(list(Path('./data/stocks/Stocks').rglob(targets[i].lower() + '.us.[tT][xX][tT]'))[0])
stocks.append(list(Path('./data/stocks').rglob('*.[cC][sS][Vv]'))[0])
stocks.append(list(Path('./data/stock_exchange').rglob('prices.[cC][sS][Vv]'))[0])

for s in stocks:
    print(s)

    # Filter out corrupted files
failed_files = []
succeeded_files = []

for i in stocks:
    try:
        df = pd.read_csv(i, sep=',')
        succeeded_files.append(i)
    except Exception:
        failed_files.append(i)

print('Succeeded:', len(succeeded_files), 'Failed:', len(failed_files))

# Read stock prices data files
dfs = {}
for sf in succeeded_files:
    df = pd.read_csv(sf, sep=',')
    dfs[sf.as_posix()] = df

# Printing out stock price data files for checking
for k, v in dfs.items():
    print(k, v.columns)

    # Permanent storage of prepared datasets
stock_dict = {}

# Prepare GAFA Stock Prices.csv
gafa_df = dfs['data/stocks/GAFA Stock Prices.csv']
# The GAFA database refers to comanpies' stocks as the companies' names,
# we have to map the companies'names onto their stocks
co_to_stock_mapping = {
    'Amazon': 'AMZN',
    'Apple': 'AAPL',
    'Facebook': 'FB',
    'Google': 'GOOGL'
}
# Convert string date to unix timestamp
gafa_new = list(
    map(
        lambda x:
        [
            x[0],  # Stock
            datetime.strptime(x[1], '%d/%m/%Y').timestamp(),  # Timestamp
            x[2],  # Open
            x[3],  # High
            x[4],  # Low
            x[5],  # Close
            x[7]  # Volume
        ],
        gafa_df.values
    )
)
gafa_dict = {}
# Create new dataframe
gafa_df = pd.DataFrame(gafa_new, columns=['Stock', 'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
# Filter on stocks, columns and sort by timestamp
for stock in sorted(set(gafa_df.values[:, 0])):
    df = gafa_df.loc[gafa_df['Stock'] == stock]
    df = df[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.sort_values(['Timestamp'])
    df = df.loc[df['Timestamp'] <= 1510272000.0]
    df = df.dropna()
    df = df.reset_index(drop=True)
    # Save our new dataframe to a dictionary
    gafa_dict[co_to_stock_mapping.get(stock)] = df

# Prepare prices.csv
prices_df = dfs['data/stock_exchange/prices.csv']


# 'cuz using only one format is too mainstream
def shitty_date_format_converter(x):
    for fmt in ('%Y-%m-%d', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.strptime(x, fmt)
        except ValueError:
            pass


# Convert string date to unix timestamp
prices_new = list(
    map(
        lambda x:
        [
            x[1],  # Stock
            shitty_date_format_converter(x[0]).timestamp(),  # TimeStamp
            x[2],  # Open
            x[5],  # High
            x[4],  # Low
            x[3],  # Close
            x[6]  # Volume
        ],
        prices_df.values
    )
)

prices_dict = {}
# Create new dataframe
prices_df = pd.DataFrame(prices_new, columns=['Stock', 'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
# Filter on stocks, columns and sort by timestamp
for stock in sorted(set(prices_df.values[:, 0]).intersection(set(targets))):
    df = prices_df.loc[prices_df['Stock'] == stock]
    df = df[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.sort_values(['Timestamp'])
    df = df.loc[df['Timestamp'] <= 1510272000.0]
    df = df.dropna()
    df = df.reset_index(drop=True)
    # Save our new dataframe to a dictionary
    prices_dict[stock] = df

# Prepare single stock datasets

# Collect names of single stock files
txt_databases = []

for k in dfs.keys():
    if k.endswith('.us.txt'):
        txt_databases.append(k)

for f in txt_databases:
    df = dfs[f]
    # Convert string date to unix timestamp
    df_new = list(
        map(
            lambda x:
            [
                datetime.strptime(x[0], '%Y-%m-%d').timestamp(),  # Timestamp
                x[1],  # Open
                x[2],  # High
                x[3],  # Low
                x[4],  # Close
                x[5]  # Volume
            ],
            df.values
        )
    )
    df = pd.DataFrame(df_new, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    # Sort dataframe by timestamp
    df = df.sort_values(['Timestamp'])
    df = df.loc[df['Timestamp'] <= 1510272000.0]
    df = df.dropna()
    df = df.reset_index(drop=True)
    # Save our new dataframe to a dictionary
    stock_dict[f.split('/')[-1].split('.')[0].upper()] = df

# Merging dataframes
for k in stock_dict.keys():
    # Appending dataframes
    if k in gafa_dict.keys():
        stock_dict[k] = stock_dict[k].append(gafa_dict[k])
        print(k, 'was found in gafa_dict, and appended it to stock_dict[', k, ']')
    if k in prices_dict.keys():
        stock_dict[k] = stock_dict[k].append(prices_dict[k])
        print(k, 'was found in prices_dict, and appended it to stock_dict[', k, ']')
    # Droping duplicates
    stock_dict[k] = stock_dict[k].drop_duplicates('Timestamp')
    stock_dict[k] = stock_dict[k].sort_values(['Timestamp'])
    stock_dict[k] = stock_dict[k].reset_index(drop=True)

# Setting temp dicts to None to prevent accidental use
gafa_dict = None
prices_dict = None
