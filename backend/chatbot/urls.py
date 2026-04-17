from django.urls import path
from .views import chat, skin, stt, tts

urlpatterns = [
    path('',      chat),
    path('stt/',  stt),
    path('tts/',  tts),
    path('skin/', skin), 
]