import re
from sklearn import naive_bayes
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder as Le, OneHotEncoder
import pandas as pandas
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.preprocessing.text import Tokenizer
import tensorflow
from sklearn.preprocessing import StandardScaler


data = pandas.read_csv('twitter_training.csv', delimiter=',', quoting=1)

source = data.iloc[0:, 1].values
tweets = data.iloc[2:, 3].values
sentiment = data.iloc[1:, 2].values


def clean_data(tweets):
  corpus = []
  for i in range(0, len(tweets)):
    review = re.sub(r"[^a-zA-Z0-9 ]", "", str(tweets[i]))
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    # review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)
  return corpus


tweets = clean_data(tweets[0:1500])


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1000)
tweets = cv.fit_transform(tweets).toarray()


x_train, x_test, y_train, y_test = train_test_split(tweets, sentiment[0:1500], test_size=0.2, random_state=0)


Le = Le()
def encode_y(y_var):
   y_var = Le.fit_transform(y_var)
   return y_var


y_train = encode_y(y_train)
y_test = encode_y(y_test)

print(y_train.shape)
print(x_train.shape)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
acc = accuracy_score(y_test, y_pred)
print(acc)

