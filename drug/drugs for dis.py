import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

df = pd.read_csv('E:\\Data Science\\Training\\Datasets\\Drug_Data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].dt.year

df.nunique()
df = df.dropna()
df = df.drop_duplicates(subset=['drugName'])


lv = TfidfVectorizer(max_features=3000,stop_words='english')
tag = df['Prescribed_for']
v = lv.fit_transform(tag).toarray()
similarity = cosine_similarity(v)

def recommendation(desieas):
    index = df[df['Prescribed_for'] == desieas].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:5]:
        print(df.iloc[i[0]])
    

# recommend = joblib.load('recommendation.sav')

recommendation('Birth Control')
