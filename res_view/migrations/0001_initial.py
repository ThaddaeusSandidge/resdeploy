# Generated by Django 4.2.6 on 2023-12-07 02:12

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ClimateData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('stn_id', models.CharField(max_length=10)),
                ('datetime', models.DateTimeField()),
                ('avgt', models.FloatField()),
                ('pcpn', models.FloatField()),
            ],
        ),
    ]
