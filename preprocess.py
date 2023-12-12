import pandas as pd
import psycopg2 as DB
from dotenv import load_dotenv
import os

if __name__ == '__main__':
    # Connect to the database
    config=load_dotenv()
    dsn='user={} host=localhost password={} dbname={}'.format(os.environ['PG_USER'], os.environ['PG_PWD'],os.environ['PG_DB'])
    conn = DB.connect(dsn)

    # Get the station data
    with conn:
        with conn.cursor() as curs:
            curs.execute("SELECT stn_id FROM meta_final;")
            rows=curs.fetchall()

    # Pefrom ML on each station
    for stn_id in rows:
        with conn:
            with conn.cursor() as curs:
                # select where stn id = stn
                curs.execute("SELECT datetime, storage_value, elev FROM res_data_final WHERE stn_id = %s;", (stn_id,))
                res=curs.fetchall()
                curs.execute("SELECT datetime, avgt, pcpn FROM clim_data WHERE stn_id = %s;", (stn_id,))
                clim=curs.fetchall()
                
        # convert to dataframes
        df_res = pd.DataFrame(res, columns=['datetime', 'storage_value', 'elev'])
        df_clim = pd.DataFrame(clim, columns=['datetime', 'avgt', 'pcpn'])

        # merge dataframes on datetime
        df = pd.merge(df_res, df_clim, on='datetime')
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Filter data for the training set (2010-2021)
        train_data = df[df['datetime'].dt.year <= 2021]

        # Filter data for testing set (2022-2023)
        test_data = df[df['datetime'].dt.year >=  2022]

        # Add a column for days since precipitation
        df['day_since_pcpn'] = 0
        days_since_pcpn = 0
        for index, row in df.iterrows():
            if row['pcpn'] > 0:
                days_since_pcpn = 0
            else:
                days_since_pcpn += 1
            df.at[index, 'day_since_pcpn'] = days_since_pcpn
        
        # Add columns for storage values of previous 14 days
        num_previous_days = 14
        for i in range(1, num_previous_days + 1):
            df[f'storage-{i}'] = df['storage_value'].shift(i)

        for i in range(1, num_previous_days + 1):
            df[f'storage-{i}'].fillna(df['storage_value'], inplace=True)


        # Filter data for the training set (2010-2021)
        train_data = df[df['datetime'].dt.year <= 2021]

        # Filter data for testing set (2022-2023)
        test_data = df[df['datetime'].dt.year >=  2022]

        train_data.to_csv('train_data-{}.csv'.format(stn_id[0]), index=False, header=True)
        test_data.to_csv('test_data-{}.csv'.format(stn_id[0]), index=False, header=True)

    conn.close()