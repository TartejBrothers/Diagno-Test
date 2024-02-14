from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, include
from chat.oldviews import index, chatbot_view

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", index, name="index"),
    path("chatbot", chatbot_view, name="chatbot"),
    path("accounts/", include("accounts.urls")),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
