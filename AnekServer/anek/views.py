from rest_framework import viewsets
# from rest_framework import permissions
from anek.serializers import AnekSerializer
from anek.models import Anek


class AnekViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows Anek to be viewed or edited.
    """
    queryset = Anek.objects.all()
    serializer_class = AnekSerializer
    # permission_classes = [permissions.IsAuthenticated]
