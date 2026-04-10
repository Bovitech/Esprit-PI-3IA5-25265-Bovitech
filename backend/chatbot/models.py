from django.db import models

class Conversation(models.Model):
    session_id = models.CharField(max_length=255)
    role = models.CharField(max_length=10) 
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.session_id} - {self.role}"