import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
from models import LSTMModel, TransformerModel
from backtesting import Backtester
from portfolio_optimization import PortfolioOptimizer
from data_acquisition import fetch_stock_data, fetch_macro_data, fetch_sentiment_data
import plotly.graph_objects as go

# Inizializzazione dell'applicazione Dash
app = dash.Dash(__name__)

# Layout dell'applicazione
app.layout = html.Div([
    html.H1('Ultimate Stock Market Simulator'),

    # Input per il simbolo del titolo azionario
    dcc.Input(id='ticker-input', value='AAPL', type='text', placeholder="Inserisci il simbolo dell'azione", style={'width': '200px'}),

    # Dropdown per selezionare l'intervallo di previsione
    dcc.Dropdown(
        id='projection-range',
        options=[
            {'label': '1 mese', 'value': '1M'},
            {'label': '6 mesi', 'value': '6M'},
            {'label': '1 anno', 'value': '1Y'},
            {'label': '5 anni', 'value': '5Y'},
            {'label': '10 anni', 'value': '10Y'},
        ],
        value='1M',
        style={'width': '200px'}
    ),
    
    # Dropdown per scegliere il modello di previsione
    dcc.Dropdown(
        id='model-selection',
        options=[
            {'label': 'LSTM', 'value': 'LSTM'},
            {'label': 'Transformer', 'value': 'Transformer'},
        ],
        value='LSTM',
        style={'width': '200px'}
    ),
    
    # Bottone per eseguire la simulazione
    html.Button('Simula', id='simulate-button', n_clicks=0),
    
    # Grafico per mostrare i risultati della previsione
    dcc.Graph(id='prediction-graph'),

    # Backtesting e ottimizzazione del portafoglio
    dcc.Graph(id='backtest-graph'),
    dcc.Graph(id='portfolio-optimization-graph'),

    # Informazioni aggiuntive e variabili macroeconomiche
    dcc.Markdown(id='macro-info'),
    dcc.Markdown(id='sentiment-info')
])

# Callback per aggiornare il grafico delle previsioni
@app.callback(
    Output('prediction-graph', 'figure'),
    [Input('simulate-button', 'n_clicks'), Input('ticker-input', 'value'), Input('projection-range', 'value'), Input('model-selection', 'value')]
)
def update_graph(n_clicks, ticker, projection_range, model_type):
    if n_clicks == 0:
        return {}

    # Ottieni i dati storici e macroeconomici
    stock_data = fetch_stock_data(ticker)
    macro_data = fetch_macro_data()
    sentiment_data = fetch_sentiment_data(ticker)

    # Seleziona il modello per la previsione
    if model_type == 'LSTM':
        model = LSTMModel(stock_data)
    elif model_type == 'Transformer':
        model = TransformerModel(stock_data)
    
    # Preprocessamento e addestramento del modello
    model.preprocess_data(macro_data, sentiment_data)
    X_train, y_train = model.create_sequences()
    model.build_model()
    model.train_model(X_train, y_train)

    # Mappa per il numero di giorni in base all'intervallo di previsione
    projection_days_map = {'1M': 30, '6M': 180, '1Y': 365, '5Y': 1825, '10Y': 3650}
    projection_days = projection_days_map.get(projection_range, 30)

    # Previsione dei prezzi futuri con il parametro projection_days
    predictions = model.predict_future(projection_days)
    
    # Crea il grafico con i dati storici e le previsioni
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Prezzo Storico'))
    future_dates = pd.date_range(start=stock_data.index[-1] + pd.Timedelta(days=1), periods=projection_days, freq='D')
    fig.add_trace(go.Scatter(x=future_dates, y=predictions.flatten(), mode='lines', name='Previsioni'))
    fig.update_layout(title=f'Previsioni per {ticker} con modello {model_type}')
    
    return fig

# Callback per il backtesting delle strategie di trading
@app.callback(
    Output('backtest-graph', 'figure'),
    [Input('simulate-button', 'n_clicks'), Input('ticker-input', 'value')]
)
def update_backtest(n_clicks, ticker):
    if n_clicks == 0:
        return {}

    # Esegui il backtesting
    stock_data = fetch_stock_data(ticker)
    backtester = Backtester(stock_data)
    results = backtester.run_backtest()

    # Verifica se `results` è vuoto o nullo
    if results is None or results.empty:
        return {}  # Ritorna un grafico vuoto se non ci sono dati

    # Crea il grafico dei risultati del backtest
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results.index, y=results['Portfolio Value'], mode='lines', name='Valore del Portafoglio'))
    fig.update_layout(title=f'Backtest della strategia per {ticker}')
    
    return fig

# Callback per l'ottimizzazione del portafoglio
@app.callback(
    Output('portfolio-optimization-graph', 'figure'),
    [Input('simulate-button', 'n_clicks')]
)
def update_portfolio_optimization(n_clicks):
    if n_clicks == 0:
        return {}

    # Ottieni i dati storici delle azioni
    stock_data = fetch_stock_data('AAPL')

    # Esegui l'ottimizzazione del portafoglio
    optimizer = PortfolioOptimizer(stock_data)
    portfolio = optimizer.optimize()

    # Crea il grafico dell'ottimizzazione del portafoglio
    fig = go.Figure()
    fig.add_trace(go.Bar(x=portfolio['Asset'], y=portfolio['Weight'], name='Peso degli asset'))
    fig.update_layout(title='Ottimizzazione del Portafoglio')
    
    return fig

# Callback per visualizzare informazioni macroeconomiche e di sentiment
@app.callback(
    [Output('macro-info', 'children'), Output('sentiment-info', 'children')],
    [Input('simulate-button', 'n_clicks'), Input('ticker-input', 'value')]
)
def update_macro_sentiment_info(n_clicks, ticker):
    if n_clicks == 0:
        return '', ''

    # Ottieni dati macroeconomici e sentiment
    macro_data = fetch_macro_data()
    sentiment_data = fetch_sentiment_data(ticker)
    
    # Ritorna le informazioni come testo formattato
    macro_info = f"**Dati Macroeconomici:**\n- Inflazione: {macro_data['inflation']}\n- Tassi di Interesse: {macro_data['interest_rates']}"
    sentiment_info = f"**Dati di Sentiment per {ticker}:**\n- Sentiment Medio: {sentiment_data['average_sentiment']}\n- Positività: {sentiment_data['positive']}%\n- Negatività: {sentiment_data['negative']}%"
    
    return macro_info, sentiment_info

# Esegui l'applicazione Dash
if __name__ == '__main__':
    app.run_server(debug=True)
