from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, include
from accounts import urls
from models.views import index, lung_index

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", index, name="index"),
    # path("chatbot", chatbot_view, name="chatbot"),
    path("accounts/", include("accounts.urls")),
    path("lung_index", lung_index, name="lung_index"),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
