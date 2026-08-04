from django.utils import timezone

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

class BorrowingAdmin(admin.ModelAdmin):
    list_display = ('user', 'book', 'status', 'borrow_date', 'due_date')
    list_filter = ('status',)
    actions = ['approve_borrowing', 'reject_borrowing', 'approve_return']

    def approve_borrowing(self, request, queryset):
        for borrowing in queryset.filter(status='pending'):
            borrowing.status = 'approved'
            borrowing.book.available -= 1
            borrowing.book.save()
            borrowing.save()
        self.message_user(request, "Selected borrowings have been approved.")
    approve_borrowing.short_description = "Approve selected borrowings"

    def approve_return(self, request, queryset):
        for borrowing in queryset.filter(status='returned_pending'):
            borrowing.status = 'returned'
            borrowing.return_date = timezone.now()
            borrowing.book.available += 1
            borrowing.book.save()
            borrowing.save()
        self.message_user(request, "Selected returns approved.")

    approve_return.short_description = "Approve Return Requests"


admin.site.register(Borrowing, BorrowingAdmin)
admin.site.register(Book, BookAdmin)
admin.site.register(Category)
admin.site.register(ClassificationFeedback)  # So you can view feedbacks in admin

