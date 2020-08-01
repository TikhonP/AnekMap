from django.urls import include, path
from . import views
from rest_framework import routers

router = routers.DefaultRouter()
router.register(r'anek', views.AnekViewSet)
router.register(r'tag', views.TagViewSet)

urlpatterns = [
    path('', include(router.urls)),
    # path('getAneksRange', views.getAneksForCanvas, name='getAneksRange'),
    path('getallanekswithlables', views.getAllAneksWithLabels, name='getallanekswithlables'),
    path('search/', views.search, name='search'),
]
