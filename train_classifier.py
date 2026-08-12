import os
import django
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from collections import Counter

# Setup Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "library_system.settings")
django.setup()

from library.models import Book, ClassificationFeedback


def clean_text(text):
    if not isinstance(text, str):
        return ""
    return text.lower().strip()


def train_model():
    print("🚀 Starting model training...")

    # 1. Collect data from database
    books = Book.objects.exclude(category__isnull=True).exclude(category__name="")

    texts = []
    labels = []

    for book in books:
        text = f"{book.title} {book.description or ''}"
        texts.append(clean_text(text))
        labels.append(book.category.name)

    print(f"📚 Loaded {len(texts)} books from database")

    # 2. Add feedback corrections
    feedbacks = ClassificationFeedback.objects.all()
    feedback_count = 0

    for fb in feedbacks:
        text = f"{fb.book.title} {fb.book.description or ''}"
        texts.append(clean_text(text))
        labels.append(fb.corrected_category)
        feedback_count += 1

    print(f"✅ Added {feedback_count} corrections from feedback")

    if len(texts) < 5:
        print("❌ Not enough data to train. Please add more books.")
        return

    # Show class distribution
    print("\n📊 Category distribution:")
    for cat, count in Counter(labels).most_common():
        print(f"  - {cat}: {count}")

    # 3. Create improved pipeline
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            stop_words='english'
        )),
        ('clf', LinearSVC(
            class_weight='balanced',
            max_iter=5000,
            random_state=42
        ))
    ])

    # 4. Train / Test split (without stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    print("\n🧠 Training model...")
    model.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n🎯 New Accuracy: {accuracy * 100:.2f}%")
    print("\n📊 Detailed Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 6. Save model
    joblib.dump(model, 'book_classifier.pkl')
    print("\n✅ Updated model saved as 'book_classifier.pkl'!")

    vectorizer = model.named_steps['tfidf']
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("✅ Vectorizer also saved!")


if __name__ == "__main__":
    train_model()