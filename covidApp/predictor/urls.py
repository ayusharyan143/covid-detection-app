
from django.contrib import admin
from django.urls import path
from predictor import views

urlpatterns = [
    path('', views.index),
    path('admin/', admin.site.urls),
]
