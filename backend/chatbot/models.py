from django.db import models


class Conversation(models.Model):
    session_id = models.CharField(max_length=255, db_index=True)
    role       = models.CharField(max_length=20)   # "user" | "assistant"
    message    = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"[{self.session_id}] {self.role}: {self.message[:60]}"


class ConversationSummary(models.Model):
    session_id = models.CharField(max_length=255, db_index=True)
    summary    = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Summary [{self.session_id}] @ {self.created_at:%Y-%m-%d %H:%M}"