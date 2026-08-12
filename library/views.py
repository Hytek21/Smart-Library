from datetime import timedelta
from django.utils import timezone

from django.views.generic import ListView
from django.db.models import Q
from .models import Book
from .models import Borrowing
from django.views.generic import DetailView
from django.contrib.auth.views import LoginView
from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import ListView
from .models import Book, Borrowing
from django.shortcuts import get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.views.generic import CreateView
from django.contrib.auth.forms import UserCreationForm
from django.urls import reverse_lazy
from django.contrib.messages.views import SuccessMessageMixin
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


@login_required
def borrow_book(request, pk):
    book = get_object_or_404(Book, pk=pk)

    if book.available <= 0:
        messages.error(request, '❌ This book is currently not available.')
        return redirect('book_detail', pk=pk)

    # Create pending request
    Borrowing.objects.create(
        user=request.user,
        book=book,
        due_date=timezone.now() + timedelta(days=14),
        status='pending'
    )

    messages.success(request,f'✅ Borrow request for "{book.title}" has been submitted. Waiting for librarian approval.')
    return redirect('my_library')


class MyLibraryView(LoginRequiredMixin, ListView):
    model = Borrowing
    template_name = 'library/my_library.html'
    context_object_name = 'borrowings'

    def get_queryset(self):
        return Borrowing.objects.filter(
            user=self.request.user,
            return_date__isnull=True
        ).order_by('due_date')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        borrowings = context['borrowings']

        context['pending_borrowings'] = borrowings.filter(status='pending')
        context['active_borrowings'] = borrowings.filter(status='approved')
        context['returned_pending'] = borrowings.filter(status='returned_pending')
        context['rejected_borrowings'] = borrowings.filter(status='rejected')

        return context


class BookDetailView(DetailView):
    model = Book
    template_name = 'library/book_detail.html'
    context_object_name = 'book'

class HomeView(TemplateView):
    template_name = 'library/home.html'

class RegisterView(SuccessMessageMixin, CreateView):
    form_class = UserCreationForm
    template_name = 'registration/register.html'
    success_url = reverse_lazy('login')
    success_message = "Account created."


class BookListView(ListView):
    model = Book
    template_name = 'library/book_list.html'
    context_object_name = 'books'
    paginate_by = 12  # Number of books per page

    def get_queryset(self):
        queryset = super().get_queryset().order_by('title')  # Sort by title by default

        # Search functionality
        query = self.request.GET.get('q')
        if query:
            queryset = queryset.filter(
                Q(title__icontains=query) |
                Q(author__icontains=query) |
                Q(description__icontains=query)
            )
        return queryset


class HomeView(LoginRequiredMixin, TemplateView):
    template_name = 'library/home.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        # Updated Stats - recalculated every time
        context['total_books'] = Book.objects.count()
        context['available_books'] = Book.objects.filter(available__gt=0).count()
        context['borrowed_books'] = Borrowing.objects.filter(
            user=self.request.user,
            status= 'approved',
            return_date__isnull=True
        ).count()

        # === ML-Powered Recommendations ===
        try:
            vectorizer = joblib.load('tfidf_vectorizer.pkl')

            user_borrowings = Borrowing.objects.filter(user=self.request.user)

            if user_borrowings.exists():
                borrowed_texts = [
                    f"{b.book.title} {b.book.description or ''}"
                    for b in user_borrowings
                ]
                borrowed_vec = vectorizer.transform(borrowed_texts).mean(axis=0)

                all_books = list(Book.objects.all())
                all_texts = [f"{book.title} {book.description or ''}" for book in all_books]
                all_vec = vectorizer.transform(all_texts)

                similarities = cosine_similarity(borrowed_vec, all_vec).flatten()

                borrowed_ids = set(user_borrowings.values_list('book_id', flat=True))

                recommended = []
                for idx in similarities.argsort()[-12:][::-1]:
                    book = all_books[idx]
                    if book.id not in borrowed_ids:
                        recommended.append(book)
                    if len(recommended) >= 6:
                        break

                context['recommended_books'] = recommended
                context['recommendation_type'] = "Based on your borrowing history"
            else:
                context['recommended_books'] = Book.objects.all()[:6]
                context['recommendation_type'] = "Popular Books"

        except Exception:
            # Fallback
            context['recommended_books'] = Book.objects.all()[:6]
            context['recommendation_type'] = "Popular Books"

        return context

class BorrowingListView(ListView):
    model = Borrowing
    template_name = 'your_template.html'   # change if needed
    context_object_name = 'borrowings'

    def get_queryset(self):
        # Only show approved borrowings to regular users
        return Borrowing.objects.filter(status='approved')


@login_required
def request_return(request, pk):
    borrowing = get_object_or_404(Borrowing, pk=pk, user=request.user)

    if borrowing.status == 'approved':
        borrowing.status = 'returned_pending'
        borrowing.save()
        messages.success(request,
                         f'✅ Return request for "{borrowing.book.title}" has been submitted. Awaiting librarian approval.')
    elif borrowing.status == 'returned_pending':
        messages.info(request, "Return request is already pending approval.")
    else:
        messages.warning(request, "This book cannot be returned at this time.")

    return redirect('my_library')

from django.shortcuts import get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Borrowing

@login_required
def dismiss_borrowing(request, pk):
    borrowing = get_object_or_404(Borrowing, pk=pk, user=request.user)

    if borrowing.status == 'rejected':
        borrowing.delete()
        messages.success(request, "Rejected request dismissed.")
    else:
        messages.warning(request, "Only rejected requests can be dismissed.")

    return redirect('my_library')