from rest_framework import serializers
from anek.models import Anek


class AnekSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Anek
        fields = '__all__'
