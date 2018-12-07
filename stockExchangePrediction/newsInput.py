from stockPreprocessing import *

# Lets Check how our news data files look
#For this purpose we load a list with the name of the files
FilesOfNews=['data/news/abcnews-date-text.csv','data/news/articles1.csv','data/news/articles2.csv','data/news/articles3.csv','data/news/RedditNews.csv','data/news/Combined_News_DJIA.csv','data/news/data.csv']

#During iteration we write out the firs two and the last line of the files with thair length and name
for filePath in []: #FilesOfNews:
  os.sys("wc -l {filePath}")
  os.sys("head -n 2 {filePath}")
  os.sys("tail -n 1 {filePath}")
  print()
  print()

# Loading in data
import csv
import sys

#For such a big file we have to set a new limit for csv
csv.field_size_limit(sys.maxsize)

#We read in the data and print it out with it length as checking
NewsContent=[]
file='data/news/abcnews-date-text.csv'

counter=0
DatabaseContent=[]
with open(file,'r') as NewsFile:
  NewsReader=csv.DictReader(NewsFile)
  for line in NewsReader:
    if line != "":
      NewsContent.append(line)

print(NewsContent[0])
print(len(NewsContent))

# using nltk it is much easier to create the word vectors
import nltk
nltk.download('punkt')

#WordCountVectors is going to be the dictionary we are going to store all existing word and the count of their accurance
WordCountVectors={}
for name in ['apple', 'amazon', 'facebook', 'google']:
  WordCountVectors[name]={}
  for line in NewsContent:
    if(name in line['headline_text']):
      words=nltk.word_tokenize(line['headline_text'])
      for word in words:
        if(word in WordCountVectors[name].keys()):
          WordCountVectors[name][word]=WordCountVectors[name][word]+1
        else:
          WordCountVectors[name][word]=1


# as a check we are printing the word vector and the count of the words relevant for amazon
print(WordCountVectors['amazon'])
sum=0
for wordC in WordCountVectors['amazon'].values():
  sum+=wordC
print(sum)

#We have to reformat the date so we have the same format as the stock data
# datetime.strptime(x[0], '%Y-%m-%d').timestamp()

for line in NewsContent:
  line['date']=datetime.strptime(line.pop('publish_date'), '%Y%m%d').timestamp()


print(NewsContent[0])

#After this we destroy the unrelevant words
top_words = 300

from collections import Counter
word_counter = Counter((' '.join([i['headline_text'] for i in NewsContent])).split(' '))
keymap = {item[0]: i+1 for i, item in enumerate(word_counter.most_common(top_words))}

#We collect the words to have them in a per day format
news_content_parsed2 = [[i['date'], [keymap[j] for j in i['headline_text'].split(' ') if j in keymap.keys()]] for i in NewsContent]

#lets check the format
news_content_parsed2[-1]

#Then we create a dictionary so we can combine the dates belonging to the same date into one big list
wordsByDate={}
for wordOfNewWithDate in news_content_parsed2:
  if wordOfNewWithDate[0] not in wordsByDate:
        wordsByDate[wordOfNewWithDate[0]]=[]
  wordsByDate[wordOfNewWithDate[0]].extend(wordOfNewWithDate[1])

#for date in wordsByDate.keys():
#  wordsByDate[date]=sorted(wordsByDate[date])

# +++
#We eliminate all data wh2ich's date is out of the time range of the stock data

stock_dates=stock_dict['AAPL'].values[:, 0]
daily_news = {}
for k, v in wordsByDate.items():
  if min(stock_dates) < k < max(stock_dates):
    daily_news[int(k/86400)*86400] = daily_news.get(int(k/86400)*86400, []) + v

#Lets see how many words we have on each day so we can decide what size of input we should have for the network
a = []
for words in wordsByDate.values():
  a.append(len(words))

#print sizes
print(min(a), np.mean(a), max(a))

# word vectors max. (kind of...) length
review_length = 600

from pandas import DataFrame
from pandas import concat
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg




