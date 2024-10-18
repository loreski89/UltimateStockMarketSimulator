import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import yfinance as yf
import plotly.graph_objs as go

# Inizializzazione dell'app Dash
app = dash.Dash(__name__)

# Layout della dashboard
app.layout = html.Div([
    html.H1("Ultimate Stock Market Simulator"),
    
    # Campo di input per il simbolo del titolo
    html.Div([
        html.Label("Inserisci il simbolo del titolo azionario:"),
        dcc.Input(id='input-symbol', type='text', value='AAPL'),  # Simbolo di default: AAPL
        html.Button(id='submit-button', n_clicks=0, children='Analizza')
    ]),
    
    # Grafico del prezzo delle azioni
    dcc.Graph(id='stock-graph')
])

# Callback per aggiornare il grafico in base al simbolo inserito
@app.callback(
    Output('stock-graph', 'figure'),
    Input('submit-button', 'n_clicks'),
    Input('input-symbol', 'value')
)
def update_graph(n_clicks, input_symbol):
    # Scarica i dati storici per il simbolo inserito
    stock_data = yf.download(input_symbol, period='1y')
    
    # Crea il grafico del prezzo
    figure = {
        'data': [
            go.Candlestick(
                x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name=input_symbol
            )
        ],
        'layout': go.Layout(
            title=f'Prezzo Storico di {input_symbol}',
            xaxis={'title': 'Data'},
            yaxis={'title': 'Prezzo'},
        )
    }
    return figure

if __name__ == '__main__':
    app.run_server(debug=True)

