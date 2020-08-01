from django.db import models
# from django.utils import timezone


class Anek(models.Model):
    text_preview = models.CharField(max_length=100, default="Null preview")
    text = models.TextField()
    date_created = models.DateTimeField(('Date'))

    x = models.IntegerField()
    y = models.IntegerField()

    likes = models.IntegerField(null=True)
    views = models.IntegerField(null=True)

    href = models.URLField(max_length=1000, unique=True, null=True)

    def __str__(self):
        return '{}'.format(self.text[:20])


class Tag(models.Model):
    name = models.CharField(max_length=50, unique=True)
    aneks = models.ManyToManyField(Anek)

    def __str__(self):
        return self.name
