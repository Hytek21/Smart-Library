from datetime import timedelta
from django.utils import timezone

from django.views.generic import ListView
from django.db.models import Q
from .models import Book
from django.views.generic import DetailView
from django.contrib.auth.views import LoginView
from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import ListView
from .models import Borrowing
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
    if book.available > 0:
        Borrowing.objects.create(
            user=request.user,
            book=book,
            due_date=timezone.now() + timedelta(days=14)
        )
        book.available -= 1
        book.save()
        messages.success(request, f"You borrowed '{book.title}'!")
    else:
        messages.error(request, "Sorry, this book isn't available right now.")
    return redirect('book_detail', pk=pk)

class MyLibraryView(LoginRequiredMixin, ListView):
    model = Borrowing
    template_name = 'library/my_library.html'
    context_object_name = 'borrowings'

    def get_queryset(self):
        return Borrowing.objects.filter(user=self.request.user, return_date__isnull=True).order_by('due_date')

class BookListView(ListView):
    model = Book
    template_name = 'library/book_list.html'
    context_object_name = 'books'
    paginate_by = 10

    def get_queryset(self):
        queryset = super().get_queryset()
        query = self.request.GET.get('q')

        if query:
            queryset = queryset.filter(
                Q(title__icontains=query) |
                Q(author__icontains=query) |
                Q(description__icontains=query)
            )
        return queryset

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


@login_required
def return_book(request, pk):
    borrowing = get_object_or_404(Borrowing, pk=pk, user=request.user)

    if not borrowing.return_date:
        borrowing.return_date = timezone.now()
        borrowing.book.available += 1
        borrowing.book.save()
        borrowing.save()
        messages.success(request, f'✅ You have successfully returned "{borrowing.book.title}"')
    else:
        messages.info(request, "This book was already returned.")

    return redirect('my_library')


class HomeView(LoginRequiredMixin, TemplateView):
    template_name = 'library/home.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        # Basic stats
        context['total_books'] = Book.objects.count()
        context['borrowed_books'] = Borrowing.objects.filter(
            user=self.request.user, return_date__isnull=True
        ).count()
        context['available_books'] = Book.objects.filter(available__gt=0).count()

        # === ML-POWERED RECOMMENDATIONS ===
        try:
            vectorizer = joblib.load('tfidf_vectorizer.pkl')

            # Get books the user has borrowed
            user_borrowed = Borrowing.objects.filter(user=self.request.user)
            borrowed_books = [b.book for b in user_borrowed]

            if borrowed_books:
                # Create text from borrowed books
                borrowed_texts = [book.title + " " + (book.description or "") for book in borrowed_books]
                borrowed_vectors = vectorizer.transform(borrowed_texts)

                # Average the vectors
                avg_vector = np.mean(borrowed_vectors.toarray(), axis=0).reshape(1, -1)

                # Get all books for comparison
                all_books = list(Book.objects.all())
                all_texts = [book.title + " " + (book.description or "") for book in all_books]
                all_vectors = vectorizer.transform(all_texts)

                # Compute similarity
                similarities = cosine_similarity(avg_vector, all_vectors).flatten()

                # Get top 6 most similar books (excluding already borrowed)
                similar_indices = similarities.argsort()[-12:][::-1]  # take more to filter

                recommended = []
                for idx in similar_indices:
                    book = all_books[idx]
                    if book not in borrowed_books and book.available > 0:
                        recommended.append(book)
                    if len(recommended) >= 6:
                        break
            else:
                recommended = Book.objects.all()[:6]  # fallback for new users

            context['recommended_books'] = recommended

        except Exception as e:
            # Fallback if model files missing
            context['recommended_books'] = Book.objects.all()[:6]

        return context