from rest_framework import serializers
from anek.models import Anek, Tag


class AnekSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Anek
        fields = '__all__'


class TagSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Tag
        fields = '__all__'
