import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np

# Initialize the Dash app
app = dash.Dash(__name__)

# Load and prepare data (this is an example using random data for a stock simulator)
np.random.seed(42)
dates = pd.date_range(start="2020-01-01", periods=200)
prices = np.cumsum(np.random.randn(200)) + 100

df = pd.DataFrame({"Date": dates, "Price": prices})

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1("Ultimate Stock Market Simulator"),
    dcc.Graph(id="price-chart"),
    dcc.Slider(
        id="range-slider",
        min=0,
        max=len(df) - 1,
        value=len(df) - 1,
        marks={i: str(df["Date"].iloc[i].strftime('%Y-%m-%d')) for i in range(0, len(df), 20)},
        step=None
    ),
    html.Div(id="output-range-slider")
])

# Callback to update the graph based on slider input
@app.callback(
    Output("price-chart", "figure"),
    [Input("range-slider", "value")]
)
def update_graph(value):
    fig = px.line(df.iloc[:value+1], x="Date", y="Price", title="Simulated Stock Prices")
    return fig

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
# Main application logic for the dashboard
