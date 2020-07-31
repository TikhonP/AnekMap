from django.urls import include, path
from . import views
from rest_framework import routers

router = routers.DefaultRouter()
router.register(r'anek', views.AnekViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
