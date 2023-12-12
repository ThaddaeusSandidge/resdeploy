import requests
import pandas as pd
import psycopg2 as DB
import datetime as dt
from dotenv import load_dotenv
import os

def getData(state_abbrev):
    url = "https://waterservices.usgs.gov/nwis/dv/?format=rdb&stateCd={}&startDT=2014-01-01&endDT={}&parameterCd=00054&siteStatus=active".format(state_abbrev, dt.datetime.now().strftime("%Y-%m-%d"))
    df = pd.read_csv(url, sep='\t', comment='#', on_bad_lines = 'skip')
    df = df.drop(df.index[0])
    df = df.drop(columns=[df.columns[0], df.columns[4]])
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])
    df.iloc[:, 2] = pd.to_numeric(df.iloc[:, 2], errors='coerce')
    df = df.dropna(subset=[df.columns[2]])
    return df

if __name__ == '__main__':
    # Connect to the database
    config=load_dotenv()
    dsn='user={} host={} password={} dbname={}'.format(os.environ['PG_USER'], os.environ['PG_HOST'], os.environ['PG_PWD'],os.environ['PG_DB'])
    conn = DB.connect(dsn)

    # Create the data tables
    with conn:
        with conn.cursor() as curs:
            # Create a table to store meta data
            curs.execute("DROP TABLE IF EXISTS res_data")
            create_data_table = """create table res_data(
                id SERIAL UNIQUE PRIMARY KEY,
                stn_id VARCHAR(10),
                datetime TIMESTAMP,
                storage_value FLOAT
            )"""
            curs.execute(create_data_table)
            conn.commit()

    # Define the abbreviations for each state
    states = ['AZ', 'CA', 'NM']

    # Fetch and process data for each state
    for state in states:
        data = getData(state)
        to_insert_data= data.values.tolist()
        insert_data = """insert into res_data (stn_id, datetime, storage_value) values (%s, %s, %s)"""
        with conn:
            with conn.cursor() as curs:
                curs.executemany(insert_data, to_insert_data)
                conn.commit()

    conn.close()