# Generated by Django 4.2.6 on 2023-12-07 02:23

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('res_view', '0002_meta_metafinal_reservoirdata_reservoirdatafinal'),
    ]

    operations = [
        migrations.DeleteModel(
            name='ClimateData',
        ),
        migrations.DeleteModel(
            name='Meta',
        ),
        migrations.DeleteModel(
            name='MetaFinal',
        ),
        migrations.DeleteModel(
            name='ReservoirData',
        ),
        migrations.DeleteModel(
            name='ReservoirDataFinal',
        ),
    ]
