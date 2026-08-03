import pandas as pd 
import numpy as np
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report
import joblib

#load Datasets 
df_fake = pd.read_csv(r"datasets\Fake.csv")
df_true = pd.read_csv(r"datasets\True.csv")

df_fake['label'] = 1
df_true['label'] = 0

df = pd.concat([df_fake , df_true], axis=0).reset_index(drop=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
def clean_news_text(text):
    text = str(text)
    text = re.sub(r'^.*?\([Aa][Rr]|Reuters\)\s*-\s*', '', text)
    text = re.sub(r'^.*?\bReuters\b\s*-\s*', '', text)
    text = text.lower()
    text = re.sub(r'\[.*?\]|\(.*?\)', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
tqdm.pandas(desc="Cleaning Text Progress")
df['full_text'] = df['title'].fillna('') + " " + df['text'].fillna('')
df['clean_text'] = df['full_text'].progress_apply(clean_news_text)
X = df['clean_text']
y = df['label']

tfidf = TfidfVectorizer(stop_words='english',
                        max_df=0.7, ngram_range=(1, 2),
                          max_features=25000,
                          sublinear_tf=True)
X_tfidf = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

model = PassiveAggressiveClassifier(max_iter=100, random_state=42,
                                    early_stopping=True,
                                    C=.8)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['REAL', 'FAKE']))

joblib.dump(model, "Fakenews.pkl")
joblib.dump(tfidf, "Tfidf.pkl")
