from django.db import models
from django.contrib.auth.models import User

class Farm(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    farm_name = models.CharField(max_length=255)
    cow_count = models.IntegerField()
    region = models.CharField(max_length=255)

    def __str__(self):
        return f"{self.farm_name} - {self.user.username}"