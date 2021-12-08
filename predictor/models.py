from django.db import models

class DataBase(models.Model):
    sentence = models.TextField(
        blank=False,null=False
    )
    