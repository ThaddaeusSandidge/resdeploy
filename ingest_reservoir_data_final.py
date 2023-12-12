import requests
import pandas as pd
import psycopg2 as DB
import datetime as dt
from dotenv import load_dotenv
import os

def getData(stn_id):
    url = "https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites={}&startDT=2010-01-01&endDT={}&siteStatus=all&siteType=LK&parameterCd=00054,62614".format(stn_id, dt.datetime.now().strftime("%Y-%m-%d"))
    df = pd.read_csv(url, sep='\t', comment='#', on_bad_lines = 'warn')
    df = df.drop(df.index[0])
    df = df.drop(columns=[df.columns[0], df.columns[4], df.columns[6]])
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])
    df.iloc[:, 2] = pd.to_numeric(df.iloc[:, 2], errors='coerce')
    df.iloc[:, 3] = pd.to_numeric(df.iloc[:, 3], errors='coerce')
    df = df.dropna(subset=[df.columns[2]])
    df = df.dropna(subset=[df.columns[3]])
    return df

if __name__ == '__main__':
    # Connect to the database
    config=load_dotenv()
    dsn='user={} host={} password={} dbname={}'.format(os.environ['PG_USER'], os.environ['PG_HOST'], os.environ['PG_PWD'],os.environ['PG_DB'])
    conn = DB.connect(dsn)
    print(conn)

    # Create the data tables
    with conn:
        with conn.cursor() as curs:
            # Create a table to store meta data
            curs.execute("DROP TABLE IF EXISTS res_data_final")
            create_data_table = """create table res_data_final(
                id SERIAL UNIQUE PRIMARY KEY,
                stn_id VARCHAR(10),
                datetime TIMESTAMP,
                storage_value FLOAT,
                elev FLOAT
            )"""
            curs.execute(create_data_table)
            conn.commit()
    print('created table')
    # Get the station data
    with conn:
        with conn.cursor() as curs:
            curs.execute("SELECT stn_id FROM meta_final;")
            rows=curs.fetchall()
    print(rows)
    # Fetch and process data for each station
    to_insert_data=[]
    to_insert_data = []
    for stn_id in rows:
        data = getData(stn_id[0])
        to_insert_data.extend([list(row) for row in data.values])
    print(f'Length of to_insert_data: {len(to_insert_data)}')

    print('fetched data')
    insert_data = """insert into res_data_final (stn_id, datetime, storage_value, elev) values (%s, %s, %s, %s)"""
    with conn:
        with conn.cursor() as curs:
            try:
                curs.executemany(insert_data, to_insert_data)
                conn.commit()
                print('inserted')
            except Exception as e:
                print(f'Error during insertion: {e}')
    print('inserted')
    conn.close()