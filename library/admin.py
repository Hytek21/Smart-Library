from django.utils import timezone
from django.contrib import messages
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User
from .models import UserProfile
from .models import Category, Book, Borrowing, ClassificationFeedback
import joblib
from django.contrib.auth.models import Group


admin.site.unregister(Group)
from sklearn.feature_extraction.text import TfidfVectorizer  # Already installed

class UserProfileInline(admin.StackedInline):
    model = UserProfile
    can_delete = False
    verbose_name_plural = 'Profile'

class CustomUserAdmin(BaseUserAdmin):
    inlines = [UserProfileInline]
    list_display = ('username', 'email', 'get_phone', 'is_staff', 'is_active')

    def get_phone(self, obj):
        try:
            return obj.profile.phone_number
        except UserProfile.DoesNotExist:
            return "-"
    get_phone.short_description = 'Phone Number'

class BookAdmin(admin.ModelAdmin):
    list_display = ('title', 'author', 'category', 'available', 'quantity')
    list_filter = ('category',)
    search_fields = ('title', 'author')

    def save_model(self, request, obj, form, change):
        # Only auto-classify if no category is selected
        if not obj.category:
            try:
                model = joblib.load('book_classifier.pkl')

                text = f"{obj.title} {obj.description or ''}"
                predicted_category_name = model.predict([text])[0]

                # Try to find the category in the database
                category = Category.objects.filter(name__iexact=predicted_category_name).first()

                if category:
                    obj.category = category
                    messages.success(
                        request,
                        f"✅ Auto-classified as: {category.name}"
                    )
                else:
                    # Create the category if it doesn't exist
                    category = Category.objects.create(name=predicted_category_name)
                    obj.category = category
                    messages.info(
                        request,
                        f"ℹ️ New category created and assigned: {category.name}"
                    )

            except Exception as e:
                messages.warning(
                    request,
                    f"⚠️ Could not auto-classify the book. Error: {str(e)}"
                )

        super().save_model(request, obj, form, change)

class BorrowingAdmin(admin.ModelAdmin):
    list_display = ('user', 'book', 'status', 'borrow_date', 'due_date', 'return_date')
    list_filter = ('status',)
    search_fields = ('user__username', 'book__title')
    actions = ['approve_borrowing', 'reject_borrowing', 'approve_return', 'reject_return']

    def approve_borrowing(self, request, queryset):
        for borrowing in queryset.filter(status='pending'):
            borrowing.status = 'approved'
            borrowing.book.available -= 1
            borrowing.book.save()
            borrowing.save()
        self.message_user(request, "Selected borrowings have been approved.")
    approve_borrowing.short_description = "Approve selected borrowings"

    def reject_borrowing(self, request, queryset):
        updated = queryset.filter(status='pending').update(status='rejected')
        self.message_user(request, f"{updated} borrowings have been rejected.")
    reject_borrowing.short_description = "Reject selected borrowings"

    def approve_return(self, request, queryset):
        for borrowing in queryset.filter(status='returned_pending'):
            borrowing.status = 'returned'
            borrowing.return_date = timezone.now()
            borrowing.book.available += 1
            borrowing.book.save()
            borrowing.save()
        self.message_user(request, "Selected returns approved.")

    approve_return.short_description = "Approve Return Requests"

    def reject_return(self, request, queryset):
        updated = queryset.filter(status='returned_pending').update(status='approved')
        self.message_user(request, f"{updated} Return Requests have been rejected.")
    reject_return.short_description = "Reject Return Requests"


admin.site.register(Borrowing, BorrowingAdmin)
admin.site.register(Book, BookAdmin)
admin.site.register(Category)
admin.site.register(ClassificationFeedback)
admin.site.unregister(User)
admin.site.register(User, CustomUserAdmin)
admin.site.register(UserProfile)

