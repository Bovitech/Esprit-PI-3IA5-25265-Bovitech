from django.urls import path
from .views import chat, stt, tts

urlpatterns = [
    path('', chat),
    path('stt/', stt),
    path('tts/', tts),
]