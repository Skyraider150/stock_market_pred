from legacy import *

# function to create input to convolutional network
def make_timeseries_instances(timeseries, window_size):
    timeseries = np.asarray(timeseries)
    X = np.atleast_3d(np.array([timeseries[start:start + window_size] for start in range(0, timeseries.shape[0] - window_size)]))
    Y = timeseries[window_size:]
    return X, Y

# preparing the data for use in the network
news_data_temp = np.asarray([np.asarray(v) for k, v in train_news])[:-7]
news_valid_temp = np.asarray([np.asarray(v) for k, v in valid_news])[:-7]
news_test_temp = np.asarray([np.asarray(v) for k, v in test_news])[:-7]

# training data
stock_data = train_stock['AAPL'][:, 1]

news_data = np.array(news_data_temp)[:-7]
stock_data, outputs = make_timeseries_instances(stock_data, 7)
outputs = np.append(outputs[1:], 0)

news_data = np.zeros((len(news_data_temp),review_length))
i = 0
for e in news_data_temp:
  j = 0
  for k in e:
    news_data[i, j] = k
    j += 1
  i += 1
# validation data

stock_valid = valid_stock['AAPL'][:, 1]

news_valid = np.array(news_valid_temp)[:-7]
stock_valid, valids = make_timeseries_instances(stock_valid, 7)
valids = np.append(valids[1:],0)

news_valid = np.zeros((len(news_valid_temp),review_length))
i = 0
for e in news_valid_temp:
  j = 0
  for k in e:
    news_valid[i, j] = k
    j += 1
  i += 1

# test data
stock_test = test_stock['AAPL'][:, 1]

news_test = np.array(news_test_temp)[:-7]
stock_test, tests = make_timeseries_instances(stock_test, 7)
tests = np.append(tests[1:],0)

news_test = np.zeros((len(news_test_temp),review_length))
i = 0
for e in news_test_temp:
  j = 0
  for k in e:
    news_test[i, j] = k
    j += 1
  i += 1

