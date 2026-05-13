from django.urls import path
from .views import CowListCreateView

urlpatterns = [
    path('cows/', CowListCreateView.as_view()),
]