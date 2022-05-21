import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

df = pd.read_csv('F:\\Data Science\\Training\\Datasets\\Drug_Data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].dt.year

df.nunique()
df = df.dropna()
df = df.drop_duplicates(subset=['drugName'])


lv = TfidfVectorizer(max_features=3000,stop_words='english')
tag = df['drugName']
v = lv.fit_transform(tag).toarray()
similarity = cosine_similarity(v)

def recommendation(drug):
    index = df[df['drugName'] == drug].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:5]:
        print(df.iloc[i[0]])
    
joblib.dump(similarity, 'similarity.sav')
# joblib.dump(df, 'df.sav')

# recommend = joblib.load('recommendation.sav')

# recommend('Nuvigil')
