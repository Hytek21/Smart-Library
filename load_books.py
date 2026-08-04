import os
import django
import pandas as pd

print("🚀 Starting book loading script...")

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "library_system.settings")
django.setup()

print("✅ Django setup completed.")

from library.models import Category, Book

print("✅ Models imported successfully.")

# Force run
filename = 'Books_data - Sheet1.csv'

try:
    df = pd.read_csv(filename)
    print(f"✅ CSV loaded! Total rows: {len(df)}")
    print(f"Columns found: {df.columns.tolist()}")

    books_loaded = 0

    for index, row in df.iterrows():
        category_name = str(row.get('category') or 'General').strip()
        category, _ = Category.objects.get_or_create(name=category_name)

        isbn = str(row.get('bid') or f"LIB{index:05d}")

        Book.objects.get_or_create(
            isbn=isbn,
            defaults={
                'title': str(row.get('title', f"Book {index}")).strip(),
                'author': str(row.get('author', 'Unknown Author')).strip(),
                'description': f"Status: {row.get('status', 'Available')}",
                'category': category,
                'quantity': 5,
                'available': 5,
            }
        )
        books_loaded += 1

    print(f"🎉 Successfully loaded {books_loaded} books!")

except FileNotFoundError:
    print(f"❌ File '{filename}' not found!")
    print("Current folder:", os.getcwd())
except Exception as e:
    print(f"❌ Error: {e}")