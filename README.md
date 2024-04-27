# PortfolioForecaster

## Description
This is a simple portfolio optimizer which uses PyTorch to forecast prices and compute historical and predicted Efficient Frontier charts, along with plotting all stocks and their predicted prices. In order for this program to work you must make a free FinancialModelingPrep API key.

### Key Placement
![alt](https://github.com/mosharieff/PortfolioForecaster/blob/main/images/ky.png)

## Details
The primary reason I made this program was to practice my skills in full-stack development along with machine learning. Of course this program is not actually able to predict the stock market but creating it sharpened my skills in linear algebra, calculus, optimization, building a websocket server, creating a React.js front-end, and utilizing PyTorch. I also add classifications to go Long or Short, with their associated probability using a Support Vector Machine algorithm from Scikit-Learn.

## Running
Make sure to install pip (or pip3) and npm and run the requirements.txt and npm install in order to be able to run the web-application
```sh
pip install -r requirements.txt
```
```sh
npm install
```
After you have completed these steps you can simply run
```sh
python server.py
```
```sh
npm start
```

## Demo
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/_9Eq4DyL7Lo/0.jpg)](https://www.youtube.com/watch?v=_9Eq4DyL7Lo)

## Pictures
### Home/Settings Page
![alt](https://github.com/mosharieff/PortfolioForecaster/blob/main/images/home.png)
### Portfolio Curve Page
![alt](https://github.com/mosharieff/PortfolioForecaster/blob/main/images/pf.png)
### Predicted Prices Page
![alt](https://github.com/mosharieff/PortfolioForecaster/blob/main/images/ps.png)
