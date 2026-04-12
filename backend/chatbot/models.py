from django.db import models

class Conversation(models.Model):
    session_id = models.CharField(max_length=255)
    role = models.CharField(max_length=10) 
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.session_id} - {self.role}"
    
class ConversationSummary(models.Model):
    session_id = models.CharField(max_length=255, db_index=True)
    summary    = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Summary [{self.session_id}] @ {self.created_at}"