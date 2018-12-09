from testOnApple import *

test_count = 20
test_len = test_stock_input.shape[0]/test_count
test_starters = []
for i in range(test_count):
  test_starters.append(test_stock_input[int(i*test_len)])

test_results = []

for j in range(test_count):
  print('Test ', j)
  current_test = int(test_len * j)
  predictions = list(test_starters[j])

  for i in range(int(test_len)):
    stock_in = np.reshape(predictions[-7:], (1, 7, 1))
    news_in = np.reshape(test_news[current_test], (1, 600))
    prediction = model.predict([stock_in, news_in])
    predictions.append(prediction)
    current_test += 1

  test_results.append(predictions)

  plt.figure(figsize=(22, 10))
plt.plot(test_stock_output)
for j in range(test_count):
  plt.plot(list(range(int(test_len * j), int(test_len * j)+int(test_len))), test_results[j][:-7])
plt.show()

