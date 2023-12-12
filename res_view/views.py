from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import pandas as pd
import psycopg2 as DB
import datetime as dt
import os
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.subplots as sp
from plotly.offline import plot
from dotenv import load_dotenv
from django.views.generic import TemplateView
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import  RationalQuadratic
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error


# Reservoir map page
def res_map(request):
    # Pass to template
    return render(request, 'res_view/res_map.html', context = {})

# Home page
def home(request):
    return render(request, 'res_view/home.html', context={})

# Reservoir selection page
def res_select(request):
   # Connect to the database
    states = [('CA', 'California'), ('AZ', 'Arizona'), ('NM', 'New Mexico')]
    return render(request, 'res_view/res_select.html', context={'data': states})

def res_prediction(request):
    config=load_dotenv()
    dsn='user={} host={} password={} dbname={}'.format(os.environ['PG_USER'], os.environ['PG_HOST'], os.environ['PG_PWD'],os.environ['PG_DB'])
    conn = DB.connect(dsn)
    with conn:
        with conn.cursor() as curs:
            curs.execute("SELECT stn_id, stn_name FROM meta_final;")
            rows=curs.fetchall()
    stations = [{'id': row[0], 'name': row[1]} for row in rows]
    models = [('rf', 'Random Forest'), ('mlp', 'MLP Regressor'), ('kn', 'KNeighbors Regressor'), ('dt', 'Decision Tree Regressor'), ('svr', 'SVR'), ('gp', 'Guassian Process')]
    return render(request, 'res_view/res_prediction.html', context={'stations':stations, 'models':models})

def res_pred_get_current(request):
    # connect to database
    config=load_dotenv()
    dsn='user={} host={} password={} dbname={}'.format(os.environ['PG_USER'], os.environ['PG_HOST'], os.environ['PG_PWD'],os.environ['PG_DB'])
    conn = DB.connect(dsn)

    # get the data for the selected station, graph, return as json
    if request.method == "GET":
        res_id=request.GET['res_id']
        res_name=request.GET['res_name']
        data_qry = "SELECT datetime, storage_value, elev FROM res_data_final WHERE stn_id = %s"
        with conn:
            with conn.cursor() as curs:
                curs.execute(data_qry, (res_id,))
                rows = curs.fetchall()
        if rows == []:
            return JsonResponse({'no_data': "No data found for this location"})        
        # Extracting data for the graph
        datetime_values = [row[0] for row in rows]
        storage_values = [row[1] for row in rows]
        elev_values = [row[2] for row in rows]

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=datetime_values, y=storage_values, name="Storage Value"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=datetime_values, y=elev_values, name="Elevation"),
            secondary_y=True,
        )
        fig.update_layout(
            title_text="Data for Location {}".format(res_name)
        )

        fig.update_xaxes(title_text="Datetime")
        fig.update_yaxes(title_text="Storage Value", secondary_y=False)
        fig.update_yaxes(title_text="Elevation", secondary_y=True)

        # prepare to pass to frontend
        curr_div = plot(fig, output_type='div', include_plotlyjs=False)

        # return the divs to the frontend
        return JsonResponse({'curr_div': curr_div})

def res_get_model(request):

    # get the data for the selected station, graph, return as json
    if request.method == "GET":
        res_id=request.GET['res_id']
        res_name=request.GET['res_name']
        mdl_code=request.GET['model_id']
        mdl_name=request.GET['model_name']
        # Read data from CSV files
        train_data = pd.read_csv('train_data-{}.csv'.format(res_id))
        test_data = pd.read_csv('test_data-{}.csv'.format(res_id))

        # Select features and target variable
        X_train = train_data.drop(['datetime', 'storage_value'], axis=1)
        y_train = train_data['storage_value']

        X_test = test_data.drop(['datetime', 'storage_value'], axis=1)
        y_test = test_data['storage_value']

        if mdl_code == 'svr':
            # Initialize and train SVR model
            svr_model = SVR(kernel='rbf', C=100, epsilon=1, gamma=0.01)
            svr_model.fit(X_train, y_train)
            y_pred = svr_model.predict(X_test)
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            
        elif mdl_code == 'gp':
            # Define the Gaussian Process Regressor
            gp_model = GaussianProcessRegressor(alpha=10, kernel= 1**2 * RationalQuadratic(alpha=0.1, length_scale=1))
            gp_model.fit(X_train, y_train)
            y_pred = gp_model.predict(X_test)
            # Calc Metrics
            mae = mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)

        elif mdl_code == 'kn':
            # Initialize KNeighborsRegressor 
            knn_model = KNeighborsRegressor(n_neighbors=15, p=3, weights='distance') 
            knn_model.fit(X_train, y_train)
            y_pred = knn_model.predict(X_test)
            # Calculate Metrics
            mae = mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)

        elif mdl_code == 'dt':
            # Initialize DecisionTreeRegressor model
            dt_model = DecisionTreeRegressor(max_depth=20, min_samples_leaf=2, min_samples_split=5) 
            dt_model.fit(X_train, y_train)
            y_pred = dt_model.predict(X_test)
            # Calculate Metrics
            mae = mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)

        elif mdl_code == 'rf':
            # Initialize RandomForestRegressor model
            rf_model = RandomForestRegressor(max_depth=30, min_samples_leaf=2, min_samples_split=10)
            rf_model.fit(X_train, y_train)
            y_pred = rf_model.predict(X_test)
            # Calculate Metrics
            mae = mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)

        elif mdl_code == 'mlp':
            # Initialize MLPRegressor model
            mlp_model = MLPRegressor(alpha=0.001, hidden_layer_sizes=(100, 100), learning_rate='adaptive')  # You can adjust parameters as needed
            mlp_model.fit(X_train, y_train)
            y_pred = mlp_model.predict(X_test)
            # Calculate Metrics
            mae = mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
        # Create subplot with 2 rows and 1 column
        fig = sp.make_subplots(rows=2, cols=1, subplot_titles=['Forecasted Values Comparison', 'Rainfall - Actual'],
                shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing = 0.1)

        # Plot the first figure in the first subplot
        trace = go.Scatter(x=test_data['datetime'], y=y_pred, mode='lines', name=mdl_name)
        actual = go.Scatter(x=test_data['datetime'], y=y_test, mode='lines', name="Actual")
        fig.add_trace(trace, row=1, col=1)
        fig.add_trace(actual, row=1, col=1)

        # Plot the second figure in the second subplot
        bar_fig = px.bar(test_data, x='datetime', y='pcpn',
                        labels={'pcpn': 'Rainfall', 'datetime': 'Date'})
        for trace in bar_fig.data:
            fig.add_trace(trace, row=2, col=1)

        # Update layout and show the figure
        fig.add_annotation(text=f'MAE: {mae:.2f}', xref='paper', yref='paper',
                   x=0.98, y=1, showarrow=False, font=dict(size=15), bgcolor='white')
        fig.add_annotation(text=f'MAPE: {mape:.5f}%', xref='paper', yref='paper',
                   x=0.98, y=0.9, showarrow=False, font=dict(size=15), bgcolor='white')
        fig.update_layout(title_text='Forecasted Values and Rainfall (2021-2023) - {}'.format(res_name))

        # prepare to pass to frontend
        predict_div = plot(fig, output_type='div', include_plotlyjs=False)

        # return the divs to the frontend
        return JsonResponse({'predict_div': predict_div})


# Get map data for map page
def res_get_map_data(request):
    config=load_dotenv()
    dsn='user={} host={} password={} dbname={}'.format(os.environ['PG_USER'], os.environ['PG_HOST'], os.environ['PG_PWD'],os.environ['PG_DB'])
    conn = DB.connect(dsn)

    # Get the station data
    query = """
    SELECT stn_id, stn_name, lat, lon
    FROM meta
    """
    with conn:
        with conn.cursor() as curs:
            curs.execute(query)        
            rows = curs.fetchall()
    stations = []
    for row in rows:
        station = {
            'id': row[0],
            'name': row[1],
            'lat': row[2],
            'lon': row[3],
        }
        stations.append(station)

    return JsonResponse({'stations': stations})

# Get station data based on state dropdown
def res_get_stations(request):
    # connect to the database
    config=load_dotenv()
    dsn='user={} host={} password={} dbname={}'.format(os.environ['PG_USER'], os.environ['PG_HOST'], os.environ['PG_PWD'],os.environ['PG_DB'])
    conn = DB.connect(dsn)

    # get the station data based on state code and return as json
    if request.method == "GET":
        state_code = request.GET['state_code']
        query = """
            SELECT stn_id, stn_name
            FROM meta
            WHERE state_code = %s
            """
        with conn:
            with conn.cursor() as curs:
                curs.execute(query, (state_code,))        
                rows = curs.fetchall()
        return JsonResponse({'data': rows})

def res_get_data(request):
    # connect to database
    config=load_dotenv()
    dsn='user={} host={} password={} dbname={}'.format(os.environ['PG_USER'], os.environ['PG_HOST'], os.environ['PG_PWD'],os.environ['PG_DB'])
    conn = DB.connect(dsn)

    # get the data for the selected station, graph, return as json
    if request.method == "GET":
        res_id=request.GET['res_id']
        res_name=request.GET['res_name']
        data_qry = "SELECT datetime, storage_value FROM res_data WHERE stn_id = %s"
        with conn:
            with conn.cursor() as curs:
                curs.execute(data_qry, (res_id,))
                rows = curs.fetchall()
        if rows == []:
            return JsonResponse({'no_data': "No data found for this location"})        
        df = pd.DataFrame(rows)
        
        # format dataframe for analysis
        df.columns = ['datetime', 'storage_value']
        df['datetime'] = pd.to_datetime(df['datetime'])

        # filter out data not from this year for current plot
        df_current = df[(df["datetime"].dt.year >= 2023)]
        df_current = df_current.set_index('datetime')
        df_current = df_current.resample('M').mean()

        # find total monthly average
        df = df.set_index('datetime')
        df_daily = df


        # Create the daily chart with 7 day average
        df_daily['7-day MA'] = df_daily['storage_value'].rolling(window=7, min_periods=1).mean()
        df_daily['7-day MA'] = df_daily['7-day MA'].interpolate()
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Scatter(x=df_daily.index, y=df_daily['storage_value'], mode='lines', name='Reservoir Storage'))
        fig_daily.add_trace(go.Scatter(x=df_daily.index, y=df_daily['7-day MA'], mode='lines', name='7-Day Moving Average',))
        title = f'Reservoir Storage Data for {res_name}'
        fig_daily.update_layout(
            title=title,
            xaxis=dict(title='Date'),
            yaxis=dict(title='Storage Value'),
        )
        # Prepare to pass to frontend
        daily_div = plot(fig_daily, output_type='div', include_plotlyjs=False)

        df = df.groupby(df.index.month).mean()

        # Create Monthly Average bar chart
        fig_monthly = go.Figure()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        fig_monthly.add_trace(go.Bar(
            x=months,
            y=df['storage_value'], 
            name='Long-Term Avg',
            text='Long-Term Average',  
            marker_color='red',
            width=0.4  
        ))
        fig_monthly.add_trace(go.Bar(
            x=months,
            y=df_current['storage_value'], 
            name='2023 Averages',
            text='This Year (2023)', 
            marker_color='blue',
            width=0.4  
        ))
        title = f'Reservoir Storage Data for {res_name}'
        fig_monthly.update_layout(
            title=title,
            xaxis=dict(title='Month'),
            yaxis=dict(title='Storage Value'),
            barmode='group',
        )
        # prepare to pass to frontend
        monthly_div = plot(fig_monthly, output_type='div', include_plotlyjs=False)

        # return the divs to the frontend
        return JsonResponse({'monthly_div': monthly_div, 'daily_div': daily_div})

