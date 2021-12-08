from django.contrib import admin
from .models import DataBase
# Register your models here.
class ModelAdmin(admin.ModelAdmin):
    list_display = ["sentence"]
admin.site.register(DataBase,ModelAdmin)