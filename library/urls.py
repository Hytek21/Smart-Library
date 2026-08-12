from django.urls import path
from .views import BookListView, BookDetailView, HomeView, borrow_book, MyLibraryView, request_return, dismiss_borrowing

urlpatterns = [
    path('', HomeView.as_view(), name='home'),
    path('', BookListView.as_view(), name='book_list'),
    path('<int:pk>/', BookDetailView.as_view(), name='book_detail'),
    path('my-library/', MyLibraryView.as_view(), name='my_library'),
    path('borrow/<int:pk>/', borrow_book, name='borrow_book'),
    path('books/', BookListView.as_view(), name='book_list'),
    path('return-request/<int:pk>/', request_return, name='request_return'),
    path('dismiss/<int:pk>/', dismiss_borrowing, name='dismiss_borrowing'),
]