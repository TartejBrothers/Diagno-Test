from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    # Custom fields to indicate user roles
    is_patient = models.BooleanField(default=False)  
    is_doctor = models.BooleanField(default=False) 

    email = models.EmailField(unique=True)  
