# Import necessary modules from Django
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

# Import the User model from the current application
from .models import User

# Create a custom admin class for the User model, inheriting from UserAdmin
class CustomUserAdmin(UserAdmin):
    # Extend the existing fieldsets from UserAdmin
    fieldsets = (
        *UserAdmin.fieldsets,  # Include the default fieldsets from UserAdmin

        # Add a new fieldset for custom fields related to user roles
        (
            'Custom Field Heading',  # Heading for the custom fieldset
            {
                'fields': (
                    'is_patirnt', 
                    'is_doctor'   
                )
            }
        )
    )

# Register the User model with the custom admin class
admin.site.register(User, CustomUserAdmin)