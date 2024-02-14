from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, include
from accounts import urls
from models.views import index, lung_index, chatbot

urlpatterns = [
    path("admin/", admin.site.urls),
    path("brain/", index, name="index"),
    path("chatbot/", chatbot, name="chatbot"),
    path("", include("accounts.urls")),
    path("lung_index", lung_index, name="lung_index"),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
