import requests
import pandas as pd
import psycopg2 as DB
import datetime as dt
from dotenv import load_dotenv
import os
import json

def getClimateData(lat,lon):
    url='https://data.rcc-acis.org/GridData'
    params={"loc":f"{lon}, {lat}","elems":[{"name":"avgt","interval":"dly","duration":"dly"},{"name":"pcpn","interval":"dly","duration":"dly"}],"sdate":"20100101","edate":"20231031","grid":"21"}
    r=requests.post(url,data=json.dumps(params),headers={'content-type': 'application/json'}, timeout=60)
    return r.json()

if __name__ == '__main__':
    # Connect to the database
    config=load_dotenv()
    dsn='user={} host={} password={} dbname={}'.format(os.environ['PG_USER'], os.environ['PG_HOST'], os.environ['PG_PWD'],os.environ['PG_DB'])
    conn = DB.connect(dsn)

    # Create the data tables
    with conn:
        with conn.cursor() as curs:
            # Create a table to store meta data
            curs.execute("DROP TABLE IF EXISTS clim_data")
            create_data_table = """create table clim_data(
                id SERIAL UNIQUE PRIMARY KEY,
                stn_id VARCHAR(10),
                datetime TIMESTAMP,
                avgt FLOAT,
                pcpn FLOAT
            )"""
            curs.execute(create_data_table)
            conn.commit()

    with conn:
        with conn.cursor() as curs:
            curs.execute("SELECT stn_id, lat, lon FROM meta_final;")
            rows=curs.fetchall()

    to_insert_data=[]
    for stn in rows:
        data=getClimateData(stn[1],stn[2])
        for i in range(len(data['data'])):
            to_insert_data.append([stn[0],data['data'][i][0],data['data'][i][1],data['data'][i][2]])
    print("fetched")
    with conn:
        with conn.cursor() as curs:
            insert_data = """insert into clim_data (stn_id, datetime, avgt, pcpn) values (%s, %s, %s, %s)"""
            curs.executemany(insert_data, to_insert_data)
            conn.commit()

    conn.close()