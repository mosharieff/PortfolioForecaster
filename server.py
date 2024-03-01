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

class FMP:

    def __init__(self, tickers, keys):
        self.url = 'https://financialmodelingprep.com/'
        self.key = keys
        self.tickers = tickers
        self.session = requests.Session()

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

    async def fetch(self, ws):
        hold = {}
        stocks, dates = await self.get_stock_data(ws)
        for t in self.tickers:
            hold[t] = stocks[t]['adjClose'].values.tolist()
        return hold, dates.tolist()
        
        
class Data:

    def __init__(self, window=100, output=30, lr=0.005, prop=0.7, vol=60):
        self.window = window
        self.output = output
        self.lr = lr
        self.prop = prop
        self.vol = vol

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

    def split_data(self, tickers, data):
        train = {}
        test = {}
        for t in tickers:
            N = len(data[t])
            I = int(self.prop*N)
            train[t] = data[t][:I]
            test[t] = data[t][I:]
        return train, test

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

    def compute_testing(self, dataset, tickers):
        test = {}
        for t in tickers:
            data = dataset[t]
            ix, iy, iz, ia = np.array(data[-self.window:]).T.tolist()
            testing_data = [(ix + iy + iz + ia),(1,)]
            IN = torch.tensor(testing_data[0], dtype=torch.float32)
            test[t] = {'model': torch.stack((IN,)), 'lastPrice': ix[-1]}
        return test

    def extract_hist_close(self, dataset, tickers):
        close = []
        whole = []
        for t in tickers:
            whole.append(dataset[t]['Price'].values.tolist())
            close.append(whole[-1][-self.output-1:])
        return np.array(close).T, whole

    def build_close(self, x, y, tickers):
        result = {}
        x, y = x, y.T.tolist()
        for t, i, j in zip(tickers, x, y):
            result[t] = {'hist_x':list(range(len(i))),'hist_y':i, 'pred_x':list(range(len(i), len(i)+len(y[0]))), 'pred_y':j}
        return result

    def stats(self, x):
        m, n = x.shape
        mu = (1/m)*np.ones(m).dot(x)
        cv = (1/(m-1))*(x - mu).T.dot(x - mu)
        sd = np.sqrt(np.diag(cv))
        return sd, mu, cv

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


class Server:

    def __init__(self):
        self.url = 'https://financialmodelingprep.com/'
        self.key = ''
        self.session = requests.Session()

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

    def ignite(self, host='0.0.0.0', port=8080):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(websockets.serve(self.server, host, port))
        loop.run_forever()

    async def server(self, ws, path):
        print("Munch")
        def splitTicks(t):
            u = list(t.keys())
            v = [t[i] for i in u]
            return u, v
        names, symbols = self.sptickers()
        # message format) reason: init or update, payload: name+symb if init else batch
        msg = {'reason':'init', 'payload':{'names':names, 'symbols':symbols}}
        await ws.send(json.dumps(msg))
        while True:
            resp = await ws.recv()
            resp = json.loads(resp)
            tickers = resp['tickers']
            epochs, window, output, lr, prop, ma_vol = resp['params']
            epochs, window, output, ma_vol = int(epochs), int(window), int(output), int(ma_vol)
            lr, prop = float(lr), float(prop)
            names, symbols = splitTicks(tickers)
            fmp = FMP(symbols, self.key)
            data = Data(window=window, output=output, lr=lr, prop=prop, vol=ma_vol)
            msg = {'reason':'update', 'payload':'Fetching Data'}
            await ws.send(json.dumps(msg))
            df, dates = await fmp.fetch(ws)
            df = data.add_metrics(df, symbols)
            msg = {'reason':'update','payload':'Splitting Dataset'}
            await ws.send(json.dumps(msg))
            train, test = data.split_data(symbols, df)
            msg = {'reason':'update', 'payload':'Training and Testing Setup'}
            await ws.send(json.dumps(msg))
            inputs, outputs = data.prepare_training(train, symbols)
            testing = data.compute_testing(test, symbols)
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

                with torch.no_grad():
                    test_outputs = model(testing[tick]['model'])

                close.append([testing[tick]['lastPrice']] + test_outputs[-1].numpy().tolist())

            hist_close, hist_total = data.extract_hist_close(test, symbols)
            adj_close = np.array(close).T

            dates = [jj if ii == 0 or ii % 30 == 0 or ii == len(hist_total[0]) - 1 else '' for ii, jj in enumerate(dates[-len(hist_total[0]):])]

            plot_prices = data.build_close(hist_total, adj_close, symbols)

            rorB = hist_close[1:]/hist_close[:-1] - 1
            rorF = adj_close[1:]/adj_close[:-1] - 1

            sdB, muB, cvB = data.stats(rorB)
            sdF, muF, cvF = data.stats(rorF)

            riskB, returnB = data.portfolio_optimizer(rorB)
            riskF, returnF = data.portfolio_optimizer(rorF)

            cax, cay, sax, say = data.max_sharpe_and_cal(muB, cvB)
            cbx, cby, sbx, sby = data.max_sharpe_and_cal(muF, cvF)

            analyze_lines = data.draw_lines(symbols, sdB.tolist(), sdF.tolist(), muB.tolist(), muF.tolist())

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
