import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
from models import LSTMModel
from data_acquisition import fetch_stock_data

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Ultimate Stock Market Simulator'),
    dcc.Input(id='ticker-input', value='AAPL', type='text', placeholder="Inserisci il simbolo dell'azione"),
    dcc.Dropdown(
        id='projection-range',
        options=[
            {'label': '1 mese', 'value': '1M'},
            {'label': '6 mesi', 'value': '6M'},
            {'label': '1 anno', 'value': '1Y'},
            {'label': '5 anni', 'value': '5Y'},
            {'label': '10 anni', 'value': '10Y'},
        ],
        value='1M'
    ),
    html.Button('Simula', id='simulate-button'),
    dcc.Graph(id='prediction-graph')
])

@app.callback(
    Output('prediction-graph', 'figure'),
    [Input('simulate-button', 'n_clicks'), Input('ticker-input', 'value'), Input('projection-range', 'value')]
)
def update_graph(n_clicks, ticker, projection_range):
    if n_clicks is None:
        return {}
    
    # Ottieni i dati storici
    data = fetch_stock_data(ticker)
    
    # Prepara i dati e addestra il modello LSTM
    lstm_model = LSTMModel(data)
    scaled_data = lstm_model.preprocess_data()
    X_train, y_train = lstm_model.create_sequences(scaled_data)
    lstm_model.build_model()
    lstm_model.train_model(X_train, y_train)
    
    # Prevedi il futuro basato sul range selezionato
    predictions = lstm_model.predict_future(X_train[-60:])  # Prevedi 60 giorni futuri
    
    # Crea il grafico delle previsioni
    fig = {
        'data': [
            {'x': data.index, 'y': data['Close'], 'type': 'line', 'name': 'Prezzo Storico'},
            {'x': pd.date_range(start=data.index[-1], periods=60, freq='D'), 'y': predictions.flatten(), 'type': 'line', 'name': 'Previsioni'}
        ],
        'layout': {
            'title': f'Previsioni per {ticker}'
        }
    }
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
