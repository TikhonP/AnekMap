# Generated by Django 3.0.6 on 2020-07-31 23:11

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Anek',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('text', models.TextField()),
                ('date_created', models.DateTimeField(verbose_name='Date')),
                ('topic', models.CharField(choices=[('SR', 'Senior')], max_length=2, null=True)),
            ],
        ),
    ]