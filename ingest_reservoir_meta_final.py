import requests
import pandas as pd
import psycopg2 as DB
import datetime as dt
from dotenv import load_dotenv
import os

# Get the meta data
def getMeta(state_abbrev):
    url = "https://waterservices.usgs.gov/nwis/site/?format=rdb&stateCd={}&parameterCd=00054&siteStatus=active".format(state_abbrev)
    df = pd.read_csv(url, sep='\t', comment='#', on_bad_lines = 'skip')
    df = df.drop(df.index[0])
    df = df.drop(columns=['agency_cd', 'coord_acy_cd', 'alt_acy_va'])
    df['state_code'] = state_abbrev
    return df

if __name__ == '__main__':
    # Connect to the database
    config=load_dotenv()
    dsn='user={} host={} password={} dbname={}'.format(os.environ['PG_USER'], os.environ['PG_HOST'], os.environ['PG_PWD'],os.environ['PG_DB'])
    conn = DB.connect(dsn)

    # Create the data table
    with conn:
        with conn.cursor() as curs:
            # Create a table to store meta data
            curs.execute("DROP TABLE IF EXISTS meta_final")
            create_meta_table = """create table meta_final(
                stn_id VARCHAR(10) UNIQUE PRIMARY KEY,
                stn_name VARCHAR(100),
                site_type VARCHAR(10),
                lat FLOAT,
                lon FLOAT,
                lat_lon_datum VARCHAR(10),
                alt FLOAT,
                alt_datum VARCHAR(10),
                huc_cd VARCHAR(10),
                state_code VARCHAR(2)
            )"""
            curs.execute(create_meta_table)
            conn.commit()

    # Define the state abreviations for data collection
    states = ['CA']

    # Fetch and process data for each state
    for state in states:
        meta = getMeta(state)
        meta = meta[meta['site_no'].isin(['11020600', '11022100'])]
        to_insert_meta= meta.values.tolist()
        insert_meta = """insert into meta_final (stn_id, stn_name, site_type, lat, lon, lat_lon_datum, alt, alt_datum, huc_cd, state_code) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""

        with conn:
            with conn.cursor() as curs:
                curs.executemany(insert_meta, to_insert_meta)
                conn.commit()

    
    conn.close()
