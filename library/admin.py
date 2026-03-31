from django.contrib import admin
from .models import Category, Book, Borrowing, ClassificationFeedback
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer  # Already installed


class BookAdmin(admin.ModelAdmin):
    def save_model(self, request, obj, form, change):
        if change and 'category' in form.changed_data:
            # Get original (before save) to compare
            original = Book.objects.get(pk=obj.pk)
            original_cat = original.category.name if original.category else None
            new_cat = obj.category.name if obj.category else None

            if original_cat != new_cat:
                # Log the correction (predicted = original auto-filled, corrected = manual)
                ClassificationFeedback.objects.create(
                    book=obj,
                    predicted_category=original_cat or "None",
                    corrected_category=new_cat or "None"
                )



        super().save_model(request, obj, form, change)

admin.site.register(Book, BookAdmin)
admin.site.register(Category)
admin.site.register(Borrowing)
admin.site.register(ClassificationFeedback)  # So you can view feedbacks in admin

