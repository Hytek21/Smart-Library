import pandas as pd
import os
import django
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "library_system.settings")
django.setup()

from library.models import ClassificationFeedback

# 1. Load original training data (your Kaggle CSV)
df_original = pd.read_csv('data.csv')  # or whatever your file is named
df_original = df_original.dropna(subset=['title', 'summary', 'genre'])
df_original['text'] = df_original['title'] + ' ' + df_original['summary']
df_original = df_original[['text', 'genre']].rename(columns={'genre': 'category'})

# 2. Load feedback from DB (corrections)
feedbacks = ClassificationFeedback.objects.all()
if feedbacks.exists():
    feedback_data = []
    for f in feedbacks:
        if f.book.title and f.corrected_category:
            text = f.book.title + ' ' + (f.book.description or '')
            feedback_data.append({'text': text, 'category': f.corrected_category})

    df_feedback = pd.DataFrame(feedback_data)
    # Append to original
    df = pd.concat([df_original, df_feedback], ignore_index=True)
    print(f"Added {len(df_feedback)} corrections to training data.")
else:
    df = df_original
    print("No corrections yet — training on original data only.")

# 3. Vectorize & Train
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
vectorizer = TfidfVectorizer(max_features=5000, stop_words=stop_words)
X = vectorizer.fit_transform(df['text'])
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(f"New Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Save updated model
joblib.dump(clf, 'book_classifier.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Updated model saved!")