print("1. Script started...")

import os
import django
import joblib
import re
from collections import Counter

print("2. Basic imports successful")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.pipeline import Pipeline
    print("3. Scikit-learn imported successfully")
except Exception as e:
    print("ERROR importing scikit-learn:", e)
    exit()

try:
    from nltk.stem import PorterStemmer
    import nltk
    print("4. NLTK imported successfully")
except Exception as e:
    print("ERROR importing NLTK:", e)
    exit()

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

print("5. NLTK data ready")

# Setup Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "library_system.settings")
django.setup()
print("6. Django setup completed")

from library.models import Book, ClassificationFeedback
print("7. Models imported successfully")

stemmer = PorterStemmer()

def clean_and_stem(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = text.split()
    stemmed = [stemmer.stem(token) for token in tokens if len(token) > 2]
    return " ".join(stemmed)

def train_model():
    print("\n8. Starting training process...")

    books = Book.objects.filter(category__isnull=False)
    print(f"9. Found {books.count()} books with categories")

    texts = []
    labels = []

    for book in books:
        raw_text = f"{book.title} {book.author or ''} {book.description or ''}"
        texts.append(clean_and_stem(raw_text))
        labels.append(book.category.name)

    print(f"10. Processed {len(texts)} books")

    # Add feedback corrections
    feedbacks = ClassificationFeedback.objects.all()
    feedback_count = 0
    for fb in feedbacks:
        raw_text = f"{fb.book.title} {fb.book.author or ''} {fb.book.description or ''}"
        texts.append(clean_and_stem(raw_text))
        labels.append(fb.corrected_category)
        feedback_count += 1

    print(f"11. Added {feedback_count} feedback corrections")

    if len(texts) < 5:
        print("Not enough data to train.")
        return

    print("\nCategory distribution:")
    for cat, count in Counter(labels).most_common():
        print(f"   - {cat}: {count}")

    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=7000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9,
            sublinear_tf=True
        )),
        ('clf', LinearSVC(
            class_weight='balanced',
            max_iter=5000,
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    print("\n12. Training the model... Please wait...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "="*50)
    print(f"MODEL ACCURACY: {accuracy * 100:.2f}%")
    print("="*50)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save model and vectorizer
    joblib.dump(model, 'book_classifier.pkl')
    joblib.dump(model.named_steps['tfidf'], 'tfidf_vectorizer.pkl')

    print("\nModel and vectorizer saved successfully!")
    print("Training completed.")

if __name__ == "__main__":
    train_model()