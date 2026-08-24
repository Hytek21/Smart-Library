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
    paginate_by = 12

    def get_queryset(self):
        queryset = Book.objects.all().order_by('title')
        query = self.request.GET.get('q')

        self.no_exact_match = False
        self.search_query = ''
        self.best_score = 0

        if not query:
            return queryset

        query = query.strip().lower()

        try:
            vectorizer = joblib.load('tfidf_vectorizer.pkl')
            all_books = list(Book.objects.all())

            if not all_books:
                return queryset

            # ===== 1. Semantic Search =====
            book_texts = [
                f"{book.title} {book.author} {book.description or ''}"
                for book in all_books
            ]

            book_vectors = vectorizer.transform(book_texts)
            query_vector = vectorizer.transform([query])

            similarities = cosine_similarity(query_vector, book_vectors).flatten()

            # ===== 2. Flexible Keyword Matching =====
            # Split the query into words and search for partial matches
            words = query.split()
            keyword_q = Q()
            for word in words:
                if len(word) > 2:  # ignore very short words
                    keyword_q |= (
                        Q(title__icontains=word) |
                        Q(author__icontains=word) |
                        Q(description__icontains=word)
                    )

            keyword_matches = set(Book.objects.filter(keyword_q).values_list('id', flat=True))

            # ===== 3. Combine both methods =====
            scored_books = []
            for idx, book in enumerate(all_books):
                score = similarities[idx]

                # Boost score if it also matches keywords
                if book.id in keyword_matches:
                    score += 0.15

                if score > 0.05 or book.id in keyword_matches:
                    scored_books.append((book, score))

            # Sort by final score
            scored_books.sort(key=lambda x: x[1], reverse=True)

            ranked_books = [book for book, score in scored_books[:12]]

            best_score = max([s for b, s in scored_books], default=0)
            self.no_exact_match = best_score < 0.18
            self.search_query = query
            self.best_score = round(best_score * 100, 1)

            return ranked_books

        except Exception as e:
            print(f"Search error: {e}")
            # Fallback to simple flexible search
            words = query.split()
            q_objects = Q()
            for word in words:
                q_objects |= (
                    Q(title__icontains=word) |
                    Q(author__icontains=word) |
                    Q(description__icontains=word)
                )
            return queryset.filter(q_objects)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['no_exact_match'] = getattr(self, 'no_exact_match', False)
        context['search_query'] = getattr(self, 'search_query', '')
        context['best_score'] = getattr(self, 'best_score', 0)
        return context



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

        # ML-Powered Recommendations
        try:
            vectorizer = joblib.load('tfidf_vectorizer.pkl')

            # Only use APPROVED borrowings for recommendations
            user_borrowings = Borrowing.objects.filter(
                user=self.request.user,
                status='approved',
                return_date__isnull=True
            ).select_related('book')

            if user_borrowings.exists():
                # Create text from borrowed books
                borrowed_texts = [
                    f"{b.book.title} {b.book.author} {b.book.description or ''}"
                    for b in user_borrowings
                ]

                # Average vector of what the user has borrowed
                borrowed_vec = vectorizer.transform(borrowed_texts).mean(axis=0)

                # All books in the system
                all_books = list(Book.objects.all())
                all_texts = [
                    f"{book.title} {book.author} {book.description or ''}"
                    for book in all_books
                ]
                all_vec = vectorizer.transform(all_texts)

                # Calculate similarity
                similarities = cosine_similarity(borrowed_vec, all_vec).flatten()

                # Exclude books the user already borrowed
                borrowed_ids = set(user_borrowings.values_list('book_id', flat=True))

                recommended = []
                for idx in similarities.argsort()[::-1]:  # Highest similarity first
                    book = all_books[idx]
                    if book.id not in borrowed_ids and book.available > 0:
                        recommended.append(book)
                    if len(recommended) >= 6:
                        break

                context['recommended_books'] = recommended
                context['recommendation_type'] = "Based on your borrowing history"
            else:
                # Fallback when user has no approved borrowings yet
                context['recommended_books'] = Book.objects.filter(available__gt=0).order_by('?')[:6]
                context['recommendation_type'] = "Popular Books"

        except Exception as e:
            print(f"Recommendation error: {e}")
            context['recommended_books'] = Book.objects.filter(available__gt=0)[:6]
            context['recommendation_type'] = "Popular Books"

        return context

class BorrowingListView(ListView):
    model = Borrowing
    template_name = 'your_template.html'
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