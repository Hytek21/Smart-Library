import re
from datetime import timedelta
from django.utils import timezone

from django.shortcuts import render, redirect
from django.contrib.auth import login
from django.contrib.auth.models import User
from .models import UserProfile
from .forms import RegisterForm
from django.views.generic import ListView
from django.db.models import Q
from nltk import word_tokenize
from nltk.stem import PorterStemmer
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
from difflib import SequenceMatcher

def register(request):
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        email = request.POST.get('email', '').strip()
        phone = request.POST.get('phone_number', '').strip()
        password1 = request.POST.get('password1', '')
        password2 = request.POST.get('password2', '')

        print("\n===== MANUAL REGISTRATION =====")
        print(f"Username: {username}")
        print(f"Email: {email}")
        print(f"Phone: {phone}")
        print("===============================\n")

        # Basic validation
        if password1 != password2:
            return render(request, 'registration/register.html', {
                'error': 'Passwords do not match'
            })

        if User.objects.filter(username=username).exists():
            return render(request, 'registration/register.html', {
                'error': 'Username already exists'
            })

        # Create user
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password1
        )

        # Create profile
        UserProfile.objects.create(
            user=user,
            phone_number=phone
        )

        login(request, user)
        return redirect('home')

    return render(request, 'registration/register.html')



# Simple synonym dictionary for academic/library terms
SYNONYMS = {
    "ai": ["artificial intelligence", "machine learning", "ml", "deep learning"],
    "intelligent": ["intelligence", "smart", "cognitive"],
    "intelligence": ["intelligent", "smart", "ai"],
    "program": ["programming", "coding", "code", "software"],
    "programming": ["program", "coding", "code", "software development"],
    "code": ["coding", "programming", "program"],
    "data": ["dataset", "information", "database"],
    "database": ["data", "db", "sql"],
    "network": ["networking", "computer network"],
    "security": ["cybersecurity", "information security"],
    "web": ["website", "web development", "frontend", "backend"],
    "algorithm": ["algorithms", "computation"],
    "science": ["scientific"],
    "math": ["mathematics", "mathematical"],
    "history": ["historical"],
    "economy": ["economics", "economic"],
    "business": ["management", "commerce"],
}

def expand_query_with_synonyms(query):
    query = query.lower().strip()
    words = query.split()
    expanded_terms = set(words)  # include original words

    for word in words:
        if word in SYNONYMS:
            expanded_terms.update(SYNONYMS[word])

    # Return both the original query and the expanded version
    expanded_query = " ".join(expanded_terms)
    return query, expanded_query


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


stemmer = PorterStemmer()

def clean_and_stem(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = word_tokenize(text)
    stemmed = [stemmer.stem(token) for token in tokens if len(token) > 2]
    return " ".join(stemmed)

def has_whole_word_match(text, word):
    if not text:
        return False
    pattern = r'\b' + re.escape(word.lower()) + r'\b'
    return bool(re.search(pattern, text.lower()))

def similarity_score(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

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

        query = query.strip()
        original_query, expanded_query = expand_query_with_synonyms(query)
        search_words = expanded_query.split()

        try:
            all_books = list(Book.objects.all())

            # ========== 1. Exact Whole Word Match (Title / Author) ==========
            exact_matches = []
            for book in all_books:
                for word in search_words:
                    if len(word) <= 2:
                        continue
                    if has_whole_word_match(book.title, word) or has_whole_word_match(book.author, word):
                        exact_matches.append(book)
                        break

            if exact_matches:
                self.no_exact_match = False
                self.search_query = query
                return exact_matches

            # ========== 2. Fuzzy Matching (Spelling Mistakes) ==========
            fuzzy_matches = []
            for book in all_books:
                title_score = similarity_score(query, book.title)
                author_score = similarity_score(query, book.author or "")

                # Also check individual words in the title
                title_words = re.findall(r'\w+', book.title.lower())
                word_scores = [similarity_score(word, tw) for word in search_words for tw in title_words]
                best_word_score = max(word_scores) if word_scores else 0

                best = max(title_score, author_score, best_word_score)

                if best >= 0.72:  # Threshold for spelling tolerance
                    fuzzy_matches.append((book, best))

            if fuzzy_matches:
                fuzzy_matches.sort(key=lambda x: x[1], reverse=True)
                self.no_exact_match = True
                self.search_query = query
                self.best_score = round(fuzzy_matches[0][1] * 100, 1)
                return [book for book, score in fuzzy_matches[:8]]

            # ========== 3. Semantic Fallback ==========
            self.no_exact_match = True
            self.search_query = query

            vectorizer = joblib.load('tfidf_vectorizer.pkl')
            book_texts = [
                clean_and_stem(f"{book.title} {book.description or ''}")
                for book in all_books
            ]

            book_vectors = vectorizer.transform(book_texts)
            query_vector = vectorizer.transform([clean_and_stem(expanded_query)])
            similarities = cosine_similarity(query_vector, book_vectors).flatten()

            scored_books = []
            for idx, book in enumerate(all_books):
                score = similarities[idx]
                if score > 0.03:
                    scored_books.append((book, score))

            scored_books.sort(key=lambda x: x[1], reverse=True)
            ranked_books = [book for book, score in scored_books[:8]]

            best_score = max([s for b, s in scored_books], default=0)
            self.best_score = round(best_score * 100, 1)

            return ranked_books

        except Exception as e:
            print(f"Search error: {e}")
            return queryset.none()

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

            # Get user's approved borrowings
            user_borrowings = Borrowing.objects.filter(
                user=self.request.user,
                status='approved',
                return_date__isnull=True
            ).select_related('book', 'book__category')

            if user_borrowings.exists():
                # 1. Get categories the user has borrowed from
                borrowed_categories = set()
                borrowed_texts = []
                borrowed_ids = set()

                for b in user_borrowings:
                    borrowed_ids.add(b.book.id)
                    if b.book.category:
                        borrowed_categories.add(b.book.category.id)
                    borrowed_texts.append(f"{b.book.title} {b.book.description or ''}")

                # 2. Prefer books from the same categories
                candidate_books = Book.objects.filter(
                    category__id__in=borrowed_categories
                ).exclude(id__in=borrowed_ids)

                # If not enough books in the same categories, add some others
                if candidate_books.count() < 6:
                    extra_books = Book.objects.exclude(
                        id__in=borrowed_ids
                    ).exclude(category__id__in=borrowed_categories)[:10]
                    candidate_books = list(candidate_books) + list(extra_books)
                else:
                    candidate_books = list(candidate_books)

                if candidate_books and borrowed_texts:
                    # Create vectors
                    borrowed_vec = vectorizer.transform(borrowed_texts).mean(axis=0)
                    borrowed_vec = np.asarray(borrowed_vec)

                    candidate_texts = [
                        f"{book.title} {book.description or ''}"
                        for book in candidate_books
                    ]
                    candidate_vectors = vectorizer.transform(candidate_texts)

                    similarities = cosine_similarity(borrowed_vec, candidate_vectors).flatten()

                    scored = list(zip(candidate_books, similarities))
                    scored.sort(key=lambda x: x[1], reverse=True)

                    recommended = [book for book, score in scored[:6]]
                else:
                    recommended = list(candidate_books)[:6]

                context['recommended_books'] = recommended
                context['recommendation_type'] = "Based on categories you have borrowed"
            else:
                # User has no borrowings yet
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