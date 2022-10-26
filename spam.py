
from cgitb import reset
import string
import joblib
import numpy as np # Линейная алгебра
import pandas as pd # Обработка и анализ данных
from wordcloud import WordCloud # Облако тегов
from wordcloud import STOPWORDS # Останавливает слова
import nltk # Символьная и статистическая обработка естественного языка
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt 
import seaborn as sns # Статистическая визуализация данных
import scikitplot as skplt #
# import elas_1 as ec
# import elas3 
from sklearn.model_selection import train_test_split # Разделяет массивы и матрицы в рандомные train and test subsets 

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer 
# CountVectorizer - Преобразование коллекции текстовых документов в матрицу подсчета токенов
# TfidfTransformer - Преобразование матрицы отсчета в нормализованное представление tf или tf-idf

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report #класс оценок
# Accuracy Score - вычисляет точность подмножества: набор меток, предсказанных для образца
# Confusion Matrix - матрица ошибок
# Classification Report - текстовый отчет, показывающий основные показатели классификации


#import methods
from sklearn.naive_bayes import MultinomialNB 
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# import namnn as nam
from sklearn.feature_extraction.text import CountVectorizer 

countvect = CountVectorizer(ngram_range = (2,2), )



def pre_Handle():
 df = pd.read_csv("C:\Doan\SpamModelAndApi\data20210902.csv", encoding = 'utf-8',on_bad_lines = 'skip',sep = "|")
 df = df.drop(columns=['Unnamed: 0'])
 df.columns = [ 'Message','Label']
 df_labels = df['Label']
 train_set, test_set, train_label, test_label = train_test_split(df, df_labels, test_size = 0.33, random_state = 42)
 countvect = CountVectorizer(ngram_range = (2,2), )
 x_counts = countvect.fit(train_set.Message)
# preparing for training set
 x_train_df = countvect.transform(train_set.Message)
# # preparing for test set
 x_test_df = countvect.transform(test_set.Message)
 return countvect






def Get_Result(text:string):

 countvect = pre_Handle()
 print("NHẬP TIN:")
#  arr_input=[input()]
#  arr_input_2 = [elas3.es_getString()]
#  print(elas3.es_getString())
 arr_input = [text]
 arr_input_ok = countvect.transform(arr_input)

 # arr_result = nam.predict(arr_input)
 loaded_model = joblib.load("C:\Doan\SpamModelAndApi\Spam_model")
 arr_result= loaded_model.predict(arr_input_ok)

 if (1 in arr_result):
    res_str = "đây là tin nhắn spam"
 else : 
    res_str = "đây là tin nhắn bình thường"
 return(res_str)




   
 

# def main():
#     Get_Result()

# if __name__ == "__main__":
#     main()


    



  
    