from django.contrib import admin
from django.urls import path
from . views import index, diagnosis

urlpatterns = [
    path("", index ,name="index"),
    path("diagnosis/", diagnosis,name="diagnosis"),
]
