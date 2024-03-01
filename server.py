import asyncio
import websockets
import requests
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize

# This class imports the data from Financial Modeling Prep's API. It requires a free api key from their site
class FMP:

    def __init__(self, tickers, keys):
        self.url = 'https://financialmodelingprep.com/'
        self.key = keys
        self.tickers = tickers
        self.session = requests.Session()

    # This fetches historical data dating back to January 1, 2021
    async def get_stock_data(self, ws):
        endpoint = 'api/v3/historical-price-full/{}?from=2021-01-01&apikey={}'
        data = {}
        for ticker in self.tickers:
            resp = self.session.get(self.url + endpoint.format(ticker, self.key)).json()
            data[ticker] = pd.DataFrame(resp['historical'])[::-1]
            msg = {'reason':'update', 'payload':f'Loaded: {ticker}'}
            await ws.send(json.dumps(msg))
        dates = data[self.tickers[0]]['date']
        return data, dates
    
    # This function returns the stock data and extracts the adjusted close prices into a dictionary where each stock ticker is the key 
    async def fetch(self, ws):
        hold = {}
        stocks, dates = await self.get_stock_data(ws)
        for t in self.tickers:
            hold[t] = stocks[t]['adjClose'].values.tolist()
        return hold, dates.tolist()

# This class prepares the data to be trained via PyTorch
class Data:

    def __init__(self, window=100, output=30, lr=0.005, prop=0.7, vol=60):
        self.window = window # Training Window
        self.output = output # Forecasted Days
        self.lr = lr         # Learning Rate
        self.prop = prop     # Proportion of Train/Test Data
        self.vol = vol       # Lookback Period: ex. self.vol = 50 = 50 day moving average

    # This function calculates the moving averages and bollinger bands and creates a pandas dataframe out of them
    def add_metrics(self, u, tickers):
        vol = self.vol
        data_frame = {}
        for t in tickers:
            data_frame[t] = []
            ux = u[t]
            for v in range(vol, len(ux)):
                window = ux[v-vol:v]
                ma = np.mean(window)
                vx = np.std(window)
                data_frame[t].append([ux[v], ma, ma - 2*vx, ma + 2*vx])
            data_frame[t] = pd.DataFrame(data_frame[t], columns=['Price','Ma','Pos','Neg'])
        return data_frame

    # This function splits the data into a training/testing set
    def split_data(self, tickers, data):
        train = {}
        test = {}
        for t in tickers:
            N = len(data[t])
            I = int(self.prop*N)
            train[t] = data[t][:I]
            test[t] = data[t][I:]
        return train, test

    # This function prepares the data to have a input window and output window and converts the data to PyTorch tensors for training
    def prepare_training(self, dataset, tickers):

        I = {}
        O = {}

        for t in tickers:
            data = dataset[t]
            training_data = []

            for i in range(self.window, len(data)-self.output+1):
                inputs = data[i-self.window:i]
                outputs = data[i:i+self.output]
                ix, iy, iz, ia = np.array(inputs).T.tolist()
                jx, jy, jz, ja = np.array(outputs).T.tolist()
                training_data.append((ix + iy + iz + ia, jx))
               
 
            IN = [torch.tensor(item[0], dtype=torch.float32) for item in training_data]
            OUT = [torch.tensor(item[1], dtype=torch.float32) for item in training_data]

            I[t] = torch.stack(IN)
            O[t] = torch.stack(OUT)

        return I, O

    # This function tests the data on the testing set and prepares as a PyTorch tensor
    def compute_testing(self, dataset, tickers):
        test = {}
        for t in tickers:
            data = dataset[t]
            ix, iy, iz, ia = np.array(data[-self.window:]).T.tolist()
            testing_data = [(ix + iy + iz + ia),(1,)]
            IN = torch.tensor(testing_data[0], dtype=torch.float32)
            test[t] = {'model': torch.stack((IN,)), 'lastPrice': ix[-1]}
        return test

    # This function extracts close prices from the testing dataset and returns the observed and the whole price data
    def extract_hist_close(self, dataset, tickers):
        close = []
        whole = []
        for t in tickers:
            whole.append(dataset[t]['Price'].values.tolist())
            close.append(whole[-1][-self.output-1:])
        return np.array(close).T, whole

    # This function prepares the historical and predicted close prices to be plotted in Plotly.js
    def build_close(self, x, y, tickers):
        result = {}
        x, y = x, y.T.tolist()
        for t, i, j in zip(tickers, x, y):
            result[t] = {'hist_x':list(range(len(i))),'hist_y':i, 'pred_x':list(range(len(i), len(i)+len(y[0]))), 'pred_y':j}
        return result

    # This function calculates the standard deviation, mean, and covariance matrix of each stock
    def stats(self, x):
        m, n = x.shape
        mu = (1/m)*np.ones(m).dot(x)
        cv = (1/(m-1))*(x - mu).T.dot(x - mu)
        sd = np.sqrt(np.diag(cv))
        return sd, mu, cv

    # This function uses Linear Algebra and Calculus to optimize for and compute the Efficient Frontier for the historical and predicted portfolio
    def portfolio_optimizer(self, x, n=60):
        def matrix(mu, cv, r):
            cov = (2.0*cv).tolist()
            for i in range(len(mu)):
                cov[i].append(mu[i])
                cov[i].append(1.0)
            cov.append(mu.tolist() + [0, 0])
            cov.append(np.ones(len(mu)).tolist() + [0, 0])
            A = np.array(cov)
            B = np.zeros(len(mu)).tolist() + [r, 1.0]
            return np.linalg.inv(A).dot(B)[:-2]
        sd, mu, cv = self.stats(x)

        r0 = np.min(mu)
        r1 = np.max(mu)
        dR = (r1 - r0)/(n - 1)
        x, y = [], []
        for i in range(n):
            rate = r0 + i*dR
            w = matrix(mu, cv, rate)
            x.append(float(np.sqrt(w.T.dot(cv.dot(w)))))
            y.append(float(w.T.dot(mu)))
        return x, y

    # This function uses Scipy to calculate the max sharpe portfolio
    def max_sharpe_and_cal(self, mu, cov):
        def optimize():
            def objective(x):
                return -(x.T.dot(mu))/(x.T.dot(cov.dot(x)))
            def cons1(x):
                return -(sum(x) - 1)
            constraints = [{'type':'eq', 'fun':cons1}]
            x = 0.1*np.ones(len(mu))
            res = minimize(objective, x, method='SLSQP', bounds=None, constraints=constraints)
            X = res.x
            risk = X.T.dot(cov.dot(X))
            retn = X.T.dot(mu)
            return float(risk), float(retn)
        ri, rn = optimize()
        ri = float(np.sqrt(ri))
        h = (0, 1, 2)
        x, y = [], []
        for weight in h:
            x.append(weight*ri)
            y.append(weight*rn + (1 - weight)*-rn)
        return x, y, ri, rn

    # This function draws lines on the portfolio chart between each historical and each predicted stock to visualize which ones get less or more risky and which ones return more or less on average
    def draw_lines(self, tickers, sd1, sd2, mu1, mu2, n=20):
        result = {}
        for t in range(len(tickers)):
            tt = tickers[t]
            xi, xj, yi, yj = sd1[t], sd2[t], mu1[t], mu2[t]
            dx = (xj - xi)/(n - 1)
            dy = (yj - yi)/(n - 1)
            result[tt] = {'x':[],'y':[]}
            for i in range(n):
                result[tt]['x'].append(xi + i*dx)
                result[tt]['y'].append(yi + i*dy)
        return result

# This class contains a feed forward Neural Network which is used to forecast the stock prices
class NeuralNet(nn.Module):

    def __init__(self, inputs, outputs):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(inputs, 300)
        self.layer2 = nn.Linear(300, 300)
        self.layer3 = nn.Linear(300, outputs)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x

# This class contains the websocket server which is the root of the program and is responsible for taking and sending messages between the React.js client
class Server:

    def __init__(self):
        self.url = 'https://financialmodelingprep.com/'
        self.key = ''
        self.session = requests.Session()

    # This function fetches all available stocks on the NYSE and NASDAQ
    def sptickers(self):
        url = self.url + f'api/v3/stock/list?apikey={self.key}'
        resp = self.session.get(url).json()
        name, symbol = [], []
        for i in resp:
            if i['type'] == 'stock':
                if i['exchangeShortName'] == 'NASDAQ' or i['exchangeShortName'] == 'NYSE':
                    if i['name'] != '':
                        try:
                            name.append(i['name'].replace('&',''))
                            symbol.append(i['symbol'])
                        except:
                            pass
        HFT = np.array([name, symbol]).T.tolist()
        HFT = list(sorted(HFT, key=lambda u: u[0]))
        name, symbol = np.array(HFT).T.tolist()
        return name, symbol

    # This function starts the websocket server
    def ignite(self, host='0.0.0.0', port=8080):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(websockets.serve(self.server, host, port))
        loop.run_forever()

    # This is the raw server function
    async def server(self, ws, path):
        print("Munch")
        def splitTicks(t):
            u = list(t.keys())
            v = [t[i] for i in u]
            return u, v

        # Sends the React.js client the initial message with all of the stock names and symbols
        names, symbols = self.sptickers()
        msg = {'reason':'init', 'payload':{'names':names, 'symbols':symbols}}
        await ws.send(json.dumps(msg))

        # Keeps program running until its closed
        while True:
            # Waits for the inputs from React.js to arrive
            resp = await ws.recv()
            resp = json.loads(resp)
            tickers = resp['tickers']
            epochs, window, output, lr, prop, ma_vol = resp['params']
            epochs, window, output, ma_vol = int(epochs), int(window), int(output), int(ma_vol)
            lr, prop = float(lr), float(prop)
            names, symbols = splitTicks(tickers)

            # Initializes the FMP class and data class and fetches the data, and then creates the dataframe
            fmp = FMP(symbols, self.key)
            data = Data(window=window, output=output, lr=lr, prop=prop, vol=ma_vol)
            msg = {'reason':'update', 'payload':'Fetching Data'}
            await ws.send(json.dumps(msg))
            df, dates = await fmp.fetch(ws)
            df = data.add_metrics(df, symbols)
            
            # This is where the data is split into a training and testing set
            msg = {'reason':'update','payload':'Splitting Dataset'}
            await ws.send(json.dumps(msg))
            train, test = data.split_data(symbols, df)

            # This is where the inputs and outputs are built for training and converting the data into a PyTorch tensor
            msg = {'reason':'update', 'payload':'Training and Testing Setup'}
            await ws.send(json.dumps(msg))
            inputs, outputs = data.prepare_training(train, symbols)
            testing = data.compute_testing(test, symbols)
            
            # This is where the model is being trained
            msg = {'reason':'update', 'payload':'Training and Testing Model'}
            await ws.send(json.dumps(msg))
            close = []
            for tick in symbols:
                model = NeuralNet(int(data.window*4), data.output)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=data.lr)
                for epoch in range(epochs):
                    out = model(inputs[tick])
                    loss = criterion(out, outputs[tick])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if (epoch + 1) % 100 == 0:
                        outline = f'{tick} has {epochs - epoch - 1} epochs left | Loss: {loss.item():.4f}'
                        msg = {'reason':'update','payload':outline}
                        await ws.send(json.dumps(msg))

                # This is where the prices are predicted
                with torch.no_grad():
                    test_outputs = model(testing[tick]['model'])

                # This is where the predicted close prices are stored into a list 
                close.append([testing[tick]['lastPrice']] + test_outputs[-1].numpy().tolist())

            # Extracts historical prices
            hist_close, hist_total = data.extract_hist_close(test, symbols)
            adj_close = np.array(close).T

            # Converts dates to be formatted for plotting in Plotly.js
            dates = [jj if ii == 0 or ii % 30 == 0 or ii == len(hist_total[0]) - 1 else '' for ii, jj in enumerate(dates[-len(hist_total[0]):])]

            # This is the dictionary which stores each stocks historical and predicted prices to plot
            plot_prices = data.build_close(hist_total, adj_close, symbols)

            # This calculates the rate of return for the historical and predicted prices
            rorB = hist_close[1:]/hist_close[:-1] - 1
            rorF = adj_close[1:]/adj_close[:-1] - 1

            # These are where the standard deviation, mean, and covariance matrices are calculated
            sdB, muB, cvB = data.stats(rorB)
            sdF, muF, cvF = data.stats(rorF)

            # This is where the Efficient Frontier curve is optimized
            riskB, returnB = data.portfolio_optimizer(rorB)
            riskF, returnF = data.portfolio_optimizer(rorF)

            # This is where the max sharpe portfolio and capital allocation lines are computed
            cax, cay, sax, say = data.max_sharpe_and_cal(muB, cvB)
            cbx, cby, sbx, sby = data.max_sharpe_and_cal(muF, cvF)

            # This is where the lines that connect the historical and forecasted portfolio stocks are calculated
            analyze_lines = data.draw_lines(symbols, sdB.tolist(), sdF.tolist(), muB.tolist(), muF.tolist())

            # This is the final message which is pushed to the React.js front-end
            msg = {'reason':'push', 'payload':{'hist_points':{'x':sdB.tolist(),'y':muB.tolist()},
                                               'hist_curve':{'x':riskB,'y':returnB},
                                               'hist_line':{'x':cax,'y':cay},
                                               'hist_sharpe':{'x':[sax],'y':[say]},
                                               'future_points':{'x':sdF.tolist(), 'y':muF.tolist()},
                                               'future_curve':{'x':riskF, 'y': returnF},
                                               'future_line':{'x':cbx,'y':cby},
                                               'future_sharpe':{'x':[sbx],'y':[sby]},
                                               'prices':plot_prices,
                                               'trend': analyze_lines,
                                               'names':names,
                                               'tickers':symbols,
                                               'dates':dates}}
            await ws.send(json.dumps(msg))


print("Server Booted......")
client = Server()
client.ignite()
