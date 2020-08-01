from django.db import models
from django.contrib.auth import get_user_model
# from django.utils import timezone

User = get_user_model()


class Anek(models.Model):
    text = models.TextField()
    date_created = models.DateTimeField(('Date'))
    TOPICS_CHOISES = (
        ('SR', 'Senior'),
    )
    topic = models.CharField(max_length=2, choices=TOPICS_CHOISES, null=True)

    # like_count = models.IntegerField()
    # views_count = models.IntegerField()


    def __str__(self):
        return '{} | {}'.format(self.text[:20], self.topic)
