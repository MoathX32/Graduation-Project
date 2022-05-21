import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import seaborn as sns

df = pd.read_csv('dataset.csv')
severity = pd.read_csv('Symptom-severity.csv')
description = pd.read_csv('symptom_Description.csv')
precaution = pd.read_csv('symptom_precaution.csv')
# df = pd.merge(df,description,on='Disease')
# df = pd.merge(df,precaution,on='Disease')


# for i in range(len(description)):
#     description["review"][i]=description["review"][i].replace('&#039;', "`")
    
df = df.iloc[:,:11]

df = df[[ 'Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4',
       'Symptom_5', 'Symptom_6', 'Symptom_7', 'Symptom_8', 'Symptom_9',
       'Symptom_10','Disease']]


df.isna().sum()

df = df.fillna('')
df.isnull().sum()
df['content'] = df['Symptom_1']+' '+df['Symptom_2'] +' '+df['Symptom_3'] +' '+df['Symptom_4'] +' '+ df['Symptom_5'] +' '+ df['Symptom_6']+' '+df['Symptom_7'] +' '+df['Symptom_8'] +' '+ df['Symptom_9']+' '+df['Symptom_10']
df = df[["content","Disease"]]

# df.dropna(inplace=True)
import re 
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')

stem = PorterStemmer()
def stemming(content):
    content = re.sub('[^a-zA-Z]',' ',content)
    content = content.lower()
    content = content.split()
    content = [stem.stem(word) for word in content if not word in stopwords.words('english')]
    content = ' '.join(content)
    return content
df['content'] = df['content'].apply(stemming)


X = df['content'].values
y = df['Disease'].values

vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 ,random_state=44)

# model = SVC()
# model.fit(x_train, y_train)

# preds = model.predict(x_test)

# conf_mat = confusion_matrix(y_test, preds)
# df_cm = pd.DataFrame(conf_mat, index=df['Disease'].unique(), columns=df['Disease'].unique())
# print('F1-score% =', f1_score(y_test, preds, average='macro')*100, '|', 'Accuracy% =', accuracy_score(y_test, preds)*100)
# sns.heatmap(df_cm)

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# from sklearn.model_selection import RandomizedSearchCV
# rf = RandomForestClassifier()
# #you can narrow down the values as you keep training
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# max_features = ['auto', 'sqrt']
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# min_samples_split = [2, 5, 10]
# min_samples_leaf = [1, 2, 4]
# bootstrap = [True, False]

# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=42)

# rf_random.fit(x_train,y_train)
# rf_random.best_params_


#using my best parameters
model = RandomForestClassifier(bootstrap=True,
 max_depth=30,
 max_features= 'sqrt',
 min_samples_leaf= 1,
 min_samples_split= 5,
 n_estimators= 1400)

model.fit(x_train,y_train)
y_pred = model.predict(x_test)
accuracy_score(y_test,y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(conf_mat, index=df['Disease'].unique(), columns=df['Disease'].unique())
print('F1-score% =', f1_score(y_test, y_pred, average='macro')*100, '|', 'Accuracy% =', accuracy_score(y_test, y_pred)*100)
sns.heatmap(df_cm)


import joblib

joblib.dump({'severity':severity,'vectorizer':vectorizer,'description':description,'precaution':precaution,'model':model}, 'Disease model text.sav')
