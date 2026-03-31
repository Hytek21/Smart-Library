from django.urls import path
from .views import BookListView, BookDetailView, HomeView, borrow_book, MyLibraryView, return_book

urlpatterns = [
    path('', HomeView.as_view(), name='home'),
    path('', BookListView.as_view(), name='book_list'),
    path('<int:pk>/', BookDetailView.as_view(), name='book_detail'),
    path('my-library/', MyLibraryView.as_view(), name='my_library'),
    path('borrow/<int:pk>/', borrow_book, name='borrow_book'),
    path('books/', BookListView.as_view(), name='book_list'),
    path('return/<int:pk>/', return_book, name='return_book'),
]