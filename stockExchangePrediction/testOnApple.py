from newsInput import *

# +++
key = 'AAPL'

# +++
# Lets eliminate all dates that are not in the stock data

# The dates that are in the stock data
stock_dates = stock_dict[key].values[:, 0]
# The dates that are in the news data
new_dates = list(daily_news.keys())

# elimination of dates of news that are not in the stock range
for date in new_dates:
    if date not in stock_dates:
        daily_news.pop(date, None)
# Add dates of the stock data to the news data with an empty llist
for date in stock_dates:
    if date not in new_dates:
        daily_news[date] = []
# Check if they are the same length
print(len(stock_dates), len(daily_news.keys()))

# +++
# Padding stuff
dn = np.asarray([np.asarray(x[1]) for x in list(daily_news.items())])
dn = np.asarray([np.pad(x[:review_length],
                        (0, review_length - len(x[:review_length]) if len(x[:review_length]) < review_length else 0),
                        'constant') for x in dn])

train_news = dn[:len(train_stock[key])]
valid_news = dn[len(train_stock[key]):len(train_stock[key]) + len(valid_stock[key])]
test_news = dn[len(train_stock[key]) + len(valid_stock[key]):]

print(len(train_news), len(valid_news), len(test_news))

# +++
tmp = stock_dict[key]
tmp = tmp.values[:, 1]
supervised_prices = series_to_supervised(list(tmp), 7).values

print(supervised_prices.shape)

train_stock_input = np.reshape(supervised_prices[:len(train_stock[key]), :7], (-1, 7, 1))
valid_stock_input = np.reshape(
    supervised_prices[len(train_stock[key]):len(train_stock[key]) + len(valid_stock[key]), :7], (-1, 7, 1))
test_stock_input = np.reshape(supervised_prices[len(train_stock[key]) + len(valid_stock[key]):, :7], (-1, 7, 1))

train_stock_output = supervised_prices[:len(train_stock[key]), 7:]
valid_stock_output = supervised_prices[len(train_stock[key]):len(train_stock[key]) + len(valid_stock[key]), 7:]
test_stock_output = supervised_prices[len(train_stock[key]) + len(valid_stock[key]):, 7:]

# import keras
from keras import regularizers

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from keras.models import Model
from keras.models import load_model

from keras.layers import Activation
from keras.layers import Bidirectional
from keras.layers import concatenate
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import MaxPooling1D
from keras.layers import LSTM
from keras.layers import Reshape

# Stock prices branch
stock_input = Input(shape=(7, 1,), dtype='float32', name='prices_input')
stock_conv1 = Conv1D(filters=32, kernel_size=5, activation='relu')(stock_input)
stock_maxp1 = MaxPooling1D(2)(stock_conv1)
# stock_dense1 = Dense(32, activation='relu')(stock_maxp1)
# stock_dense2 = Dense(32, activation='relu')(stock_dense1)
# stock_conv2 = Conv1D(filters=1, kernel_size=5, activation='relu')(stock_maxp1)
# stock_maxp2 = MaxPooling1D()(stock_conv2)
stock_dense3 = Dense(16, activation='relu')(stock_maxp1)
stock_dense4 = Dense(64, activation='relu')(stock_dense3)

# Stock prices branch
stock_input = Input(shape=(7, 1,), dtype='float32', name='prices_input')
stock_conv1 = Conv1D(16, 5, activation='relu', padding='same')(stock_input)
stock_maxp1 = MaxPooling1D(2)(stock_conv1)
stock_conv2 = Conv1D(16, 5, activation='relu', padding='same')(stock_maxp1)
stock_maxp2 = MaxPooling1D(2)(stock_conv2)
stock_dense3 = Dense(64, activation='relu')(stock_maxp2)
stock_dense4 = Dense(64, activation='relu')(stock_dense3)

# News branch
news_input = Input(shape=(review_length,), dtype='int32', name='news_input')
news_embedding = Embedding(top_words, 64, input_length=review_length)(news_input)
news_lstm = Bidirectional(LSTM(8))(news_embedding)
news_dense = Dense(64, activation='relu')(news_lstm)
news_reshape = Reshape((1, 64,))(news_dense)

# Network trunk
conc_input = concatenate([stock_dense4, news_reshape])
conc_lstm1 = Bidirectional(LSTM(8, return_sequences=True))(conc_input)
conc_lstm2 = Bidirectional(LSTM(4))(conc_lstm1)
conc_dense1 = Dense(32, activation=('relu'))(conc_lstm2)
conc_drop1 = Dropout(0.2)(conc_dense1)
conc_dense2 = Dense(16, activation=('relu'))(conc_drop1)
conc_drop2 = Dropout(0.2)(conc_dense2)
conc_dense3 = Dense(8, activation=('relu'))(conc_drop2)
conc_drop3 = Dropout(0.2)(conc_dense3)
conc_dense4 = Dense(1, activation=('sigmoid'), name='main_output')(conc_drop3)

os.system("rm weights.hdf5")
os.system("ls")

callbacks = [
    EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=1, mode='auto'),
    ModelCheckpoint('weights.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
]

# Setting up and Training the Network
model = Model(inputs=[stock_input, news_input], outputs=[conc_dense4])

model.compile(optimizer='adam', loss='mse')

history = model.fit([train_stock_input, train_news], [train_stock_output], epochs=10000, batch_size=128,
                    callbacks=callbacks, validation_data=([valid_stock_input, valid_news], [valid_stock_output]))

os.system("ls -lh")

model = load_model('weights.hdf5')


# Train data
preds = model.predict([train_stock_input, train_news])
plt.figure(figsize=(22, 10))
plt.plot(train_stock_output[1:])
plt.plot(preds)
plt.show()

# Validation data
preds = model.predict([valid_stock_input, valid_news])
plt.figure(figsize=(22, 10))
plt.plot(valid_stock_output[1:])
plt.plot(preds)
plt.show()

# Test data
preds = model.predict([test_stock_input, test_news])
plt.figure(figsize=(22, 10))
plt.plot(test_stock_output[1:])
plt.plot(preds)
plt.show()

