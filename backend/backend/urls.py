from django.contrib import admin
from django.urls import path, include
from chatbot.views import frontend

urlpatterns = [
    path('', frontend), 
    path('admin/', admin.site.urls),
    path('chatbot/', include('chatbot.urls')),
]