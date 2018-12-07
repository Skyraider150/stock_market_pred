from testOnApple import *

#test_stock_input
#test_news

print(test_stock_input.shape, test_news.shape)
print(test_stock_input[0])
print(test_news[0])

pred_len = 21

price = test_stock_input[0]
for i in range(pred_len):
  news = np.reshape(test_news[i], (1, 600))
  pred = model.predict([np.reshape(price[-7:], (1, 7, 1)), news])
  price = np.concatenate((price, pred))

plt.plot(np.concatenate((test_stock_input[0], test_stock_input[7], test_stock_input[14])))
plt.plot(price)
plt.show()

for i in range(int((test_stock_input.shape[0]-7)/test_stock_input.shape[1])):
  model.predict([np.reshape(test_stock_input[i*7], (1, 7, 1)), np.reshape(test_news[i*7], (1, 600))])

from keras.utils.vis_utils import plot_model
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

plot_model(model, show_shapes=True, show_layer_names=True)

from IPython.display import Image, display
display(Image('model.png'))

