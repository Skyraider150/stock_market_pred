from testing import *

#
#
# ASDASDASD
#
#

#Lets unite the data of the dictionary in to one dataset of list

news_data=daily_news.values
stock_data_tmp = []
for k, df in stock_dict.items():
  stock_data_tmp.append(df.values)

# lets organize the stock data into a list of dictionaries (value per company)

stock_data=[]
size=len(stock_data_tmp)-1
for i in range(0,size):
  stock_data.append({})
  x=0
  for company in stock_dict.keys():
    stock_data[i][company]=stock_data_tmp[x][i]
    x=x+1

# Creating the datasets for the other part of the network: the news

train_news, valid_news, test_news = {},{},{}

vals = list(daily_news.items())
vs = int(len(vals) * valid_split)
ts = int(len(vals) * test_split)
train_news = vals[:vs]
valid_news = vals[vs:ts]
test_news = vals[ts:]
print('\ttrain length:', len(train_news))
print('\tvalid length:', len(valid_news))
print('\ttest  length:', len(test_news))


