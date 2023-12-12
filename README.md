# Res Tool

<table>
<tr>
<td>
  This project is an data visualization web app built on the Django framework used to analyze and predict trends in reservoir storage data. Machine Learning techniques are used to predict hindcasted reservoir storage values using reservoir and climate based time series data. Additionally, explore the real time values of surrounding reservoirs using interactive map visualizations and drop downs.
</td>
</tr>
</table>

## Demo

Here is a working live demo : https://res-tool.onrender.com

## Site

### Res Predict

Select a machine learning model to predict hindcasted reservoir levels for San Vicente Reservoir and El Capitan Reservoir and observe the difference in model prediction accuracies.
Models Used:

- Neural Network
- Gaussian Process
- SVR
- Decision Tree
- Random Forest
- Nearest Neighbor

#### Neural Network Model saw results of up to 99.8% accuracy!

### Res Select

Select state and station with dynamically populated dropdowns to view the data for each location.

### Res Map

Select locations from an interactive map to view the data forneach station.

## Built with

- Python
- PostGres
- Django
- SciKit Learn
- Pandas
- Leaflet JS
