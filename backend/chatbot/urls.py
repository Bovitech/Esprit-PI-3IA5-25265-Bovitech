from django.urls import path
from .views import chat, stt

urlpatterns = [
    path('', chat),
    path('stt/', stt),
]