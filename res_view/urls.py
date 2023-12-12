from django.urls import path
from res_view import views

urlpatterns = [
    path("", views.home, name="home"),
    path("res_select/", views.res_select, name="res_select"),
    path("res_map/", views.res_map, name="res_map"),
    path("res_prediction/", views.res_prediction, name="res_prediction"),
    path("res_get_data/", views.res_get_data, name="res_get_data"),
    path("res_get_model/", views.res_get_model, name="res_get_model"),
    path("res_pred_get_current/", views.res_pred_get_current, name="res_pred_get_current"),
    path("res_get_stations/", views.res_get_stations, name="res_get_stations"),
    path("res_get_map_data/", views.res_get_map_data, name="res_get_map_data")
]