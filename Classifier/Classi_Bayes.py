#classifier E/N used as representation to run into naive bayes method using raw text

import nltk
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from itertools import starmap
from operator import concat
from nltk.corpus import stopwords
from nltk.test.portuguese_en_fixt import setup_module
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher
from nltk.metrics import edit_distance
from nltk.metrics.distance import jaccard_distance
from nltk.metrics import masi_distance
from nltk.metrics import binary_distance
from nltk.metrics import agreement
from sklearn.metrics import confusion_matrix


nltk.download("stopwords")

tree_TEST = ET.parse('assin-ptbr-test.xml')
root_TEST = tree_TEST.getroot()

tree_TRAIN = ET.parse('assin-ptbr-train.xml')
root_TRAIN = tree_TRAIN.getroot()


#criando a lista do eixo y do set de treinamento (labels T ou H dos pares)
listalabel_TEST = list()
listalabel_TRAIN  = list()
string_H_train = list()
string_T_train = list()
metrics_TRAIN = list()

for pair in root_TRAIN.findall("pair"):
  h = pair.find('h').text
  t = pair.find('t').text

  entailment = pair.get('entailment')

  string_H_train.append((h))
  string_T_train.append((t))

  listalabel_TRAIN.append((entailment))

#criando as listas de pares H e T para testes
lista_TEST_H = list()
lista_TEST_T = list()


metricas_TEST_list = list()
for pair in root_TEST.findall("pair"):
  h = pair.find('h').text
  t = pair.find('t').text

  entailment = pair.get('entailment')

  lista_TEST_H.append((h))
  lista_TEST_T.append((t))
  listalabel_TEST.append((entailment))
  metricas_TEST = [rouge_similarity(h, t),jaro_winkler_similarity(h,t, prefix_scale=0.5),jaccard_distance(set(h),set(t)),length_similarity(h, t),lcs_similarity(h, t)]
  metricas_TEST_list.append(metricas_TEST)


metricas_TRAIN_list = list()
for pair in root_TRAIN.findall("pair"):
  h = pair.find('h').text
  t = pair.find('t').text

  string_H_train.append((h))
  string_T_train.append((t))
  metricas_TRAIN = [rouge_similarity(h, t),jaro_winkler_similarity(h,t, prefix_scale=0.5),jaccard_distance(set(h),set(t)),length_similarity(h, t),lcs_similarity(h, t)]
  metricas_TRAIN_list.append(metricas_TRAIN)

#definindo os parametros X,Y  para o metodo de aprendizagem de maquina
X_train = metricas_TRAIN_list
Y_train = listalabel_TRAIN

X_test =  metricas_TEST_list
Y_test =  listalabel_TEST


#Machine Learning method
from sklearn.naive_bayes import MultinomialNB
cls = MultinomialNB()
cls.fit(X_train, Y_train)
from sklearn.metrics import classification_report, accuracy_score
y_pred = cls.predict(X_test)


print(accuracy_score(Y_test, y_pred))
print(classification_report(Y_test, y_pred))

confusion_mat = confusion_matrix(Y_test, y_pred)
print(confusion_mat)
