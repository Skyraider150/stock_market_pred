from preprocessingAndTraining import *

from keras.layers import concatenate
from keras.models import Model
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import LSTM
from keras.layers import Reshape
from keras.layers import Bidirectional


# Stock prices branch
stock_input = Input(shape=(7, 1,), dtype='float32', name='prices_input')
stock_conv1 = Conv1D(filters=8, kernel_size=5, activation='relu')(stock_input)
stock_maxp1 = MaxPooling1D()(stock_conv1)
#stock_dense1 = Dense(32, activation='relu')(stock_maxp1)
#stock_dense2 = Dense(32, activation='relu')(stock_dense1)
#stock_conv2 = Conv1D(filters=8, kernel_size=5, activation='relu')(stock_maxp1)
#stock_maxp2 = MaxPooling1D()(stock_conv2)
stock_dense3 = Dense(16, activation='relu')(stock_maxp1)
stock_dense4 = Dense(64, activation='relu')(stock_dense3)

# News branch
news_input = Input(shape=(review_length,), dtype='int32', name='news_input')
news_embedding = Embedding(top_words, 64, input_length=review_length)(news_input)
news_lstm1 = Bidirectional(LSTM(8))(news_embedding)
news_dense = Dense(64, activation='relu')(news_lstm1)
news_reshape = Reshape((1, 64, ))(news_dense)

# Network trunk
conc_input = concatenate([stock_dense4, news_reshape])
conc_lstm1 = Bidirectional(LSTM(8, return_sequences=True))(conc_input)
conc_lstm2 = Bidirectional(LSTM(4))(conc_lstm1)
conc_dense1 = Dense(8, activation=('relu'))(conc_lstm2)
conc_dense2 = Dense(4, activation=('relu'))(conc_dense1)
conc_dense3 = Dense(4, activation=('relu'))(conc_dense2)
conc_dense4 = Dense(1, activation=('sigmoid'), name='main_output')(conc_dense3)

# Setting up and Training the Network
model = Model(inputs=[stock_input, news_input], outputs=[conc_dense4])

model.compile(optimizer='rmsprop', loss='mean_squared_error', loss_weights=[0.5])

history = model.fit([stock_data, news_data], [outputs], epochs=2, batch_size=64, validation_data=([stock_valid, news_valid], [valids]))

# Visualization of the learning
preds = model.predict([stock_test, news_test])

plt.figure(figsize=(12,8))
plt.plot(preds, color='r', label="Predictions")
plt.plot(tests, color='g', label="Targets")
plt.legend()


