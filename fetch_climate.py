# fetch_climate.py
import requests
from datetime import datetime
from django.db import models, transaction
from dotenv import load_dotenv
import os
import json

load_dotenv()

# Define your Django models here
class ClimateData(models.Model):
    stn_id = models.CharField(max_length=10)
    datetime = models.DateTimeField()
    avgt = models.FloatField()
    pcpn = models.FloatField()

    def __str__(self):
        return f'{self.stn_id} - {self.datetime}'

def get_climate_data(lat, lon):
    url = 'https://data.rcc-acis.org/GridData'
    params = {
        "loc": f"{lon}, {lat}",
        "elems": [
            {"name": "avgt", "interval": "dly", "duration": "dly"},
            {"name": "pcpn", "interval": "dly", "duration": "dly"}
        ],
        "sdate": "20100101",
        "edate": "20231031",
        "grid": "21"
    }
    r = requests.post(url, data=json.dumps(params), headers={'content-type': 'application/json'}, timeout=60)
    return r.json()

# Function to handle database interactions
def main():
    # Connect to the default database specified in your Django settings
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "reservoir.settings")
    import django
    django.setup()

    # Create the data tables
    with transaction.atomic():
        ClimateData.objects.all().delete()

        rows = ClimateData.objects.values_list('stn_id', 'lat', 'lon')

        to_insert_data = []
        for stn in rows:
            data = get_climate_data(stn[1], stn[2])
            for i in range(len(data['data'])):
                to_insert_data.append(ClimateData(
                    stn_id=stn[0],
                    datetime=datetime.strptime(data['data'][i][0], "%Y-%m-%d"),
                    avgt=data['data'][i][1],
                    pcpn=data['data'][i][2]
                ))

        ClimateData.objects.bulk_create(to_insert_data)

# Call the main function
if __name__ == '__main__':
    main()
