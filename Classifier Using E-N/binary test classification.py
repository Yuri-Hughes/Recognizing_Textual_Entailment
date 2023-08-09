# Classifier E/NE representation

import numpy as np

import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from operator import concat
from itertools import starmap
from nltk.corpus import stopwords
import nltk
from nltk.test.portuguese_en_fixt import setup_module


nltk.download("stopwords")


tree = ET.parse('assin-ptbr-test.xml')
root = tree.getroot()

tree_TRAIN = ET.parse('assin-ptbr-test.xml')
root_TRAIN = tree_TRAIN.getroot()

lista_TEST     = list()
listlabel_TEST = list()

for pair in root.findall("pair"):
  h = pair.find('h').text
  t = pair.find('t').text

  lista_TEST.append((h,t))

  entailment = pair.get('entailment')
  listlabel_TEST.append((entailment))

result_TEST = list(starmap(concat, lista_TEST))


lista_TRAIN       = list()
listalabel_TRAIN  = list()

for pair in root_TRAIN.findall("pair"):
  h = pair.find('h').text
  t = pair.find('t').text

  lista_TRAIN.append((h,t))

  entailment = pair.get('entailment')
  listalabel_TRAIN.append((entailment))

result_TRAIN = list(starmap(concat, lista_TRAIN))

X_train = result_TRAIN
X_test  = result_TEST
y_train = listalabel_TRAIN
y_test  = listlabel_TEST


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words= nltk.corpus.stopwords.words('portuguese'), max_features=1000, decode_error="ignore")
vectorizer.fit(X_train)

vectorizer.fit(X_train)
X_train_vectorized = vectorizer.transform(X_train)



from sklearn.naive_bayes import MultinomialNB
cls = MultinomialNB()
cls.fit(vectorizer.transform(X_train), y_train)

from sklearn.metrics import classification_report, accuracy_score
y_pred = cls.predict(vectorizer.transform(X_test))

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))