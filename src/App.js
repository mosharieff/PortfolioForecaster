import React, { Component, Fragment } from 'react'
import Plot from 'react-plotly.js'
import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
import 'react-tabs/style/react-tabs.css';


export default class App extends Component {
  constructor(){
    super()

    this.state = {response:null,
                  enter:'',
                  stocks:[],
                  names: [],
                  cstocks:[],
                  cnames:[],
                  tickers:{},
                  epochs:1000,
                  window:130,
                  output:30,
                  lr:0.0001,
                  prop:0.7,
                  ma_vol:50,
                  skip: 10,
                  sock:null,
                  run_state:'idle',
                  df:null}

    this.titleBar = this.titleBar.bind(this)
    this.inputChange = this.inputChange.bind(this)
    this.spInputs = this.spInputs.bind(this)
    this.AddTicker = this.AddTicker.bind(this)
    this.DisplayTicks = this.DisplayTicks.bind(this)
    this.DisplayParams = this.DisplayParams.bind(this)
    this.adjustInputs = this.adjustInputs.bind(this)
    this.fetchData = this.fetchData.bind(this)
    this.clearData = this.clearData.bind(this)
    this.plotEF = this.plotEF.bind(this)
    this.plotPrices = this.plotPrices.bind(this)
  }

  componentDidMount(){
    const socket = new WebSocket('ws://localhost:8080')
    socket.onmessage = (evt) => {
      const response = JSON.parse(evt.data)
      if(response['reason'] === 'init'){
        this.setState({ stocks: response['payload']['symbols'],
                        names: response['payload']['names']
                      })
      }
      if(response['reason'] === 'update'){
        this.setState({run_state: response['payload']})
      }
      if(response['reason'] === 'push'){
        this.setState({run_state: 'imported', df:response['payload']})
      }
    }
    this.setState({ sock: socket })
  }
  
  titleBar(title){
    const bg = 'black'
    const fg = 'yellow'
    return(
      <div style={{backgroundColor: bg, color: fg, fontSize: 50}}>
        {title}
      </div>
    )
  }

  AddTicker(evt){
    const { tickers, cnames, cstocks } = this.state
    const n = parseInt(evt.target.value)
    tickers[cnames[n]] = cstocks[n]
    this.setState({tickers: tickers})
    evt.target.value = ""
  }

  spInputs(){
    const hold = []
    const { cstocks, cnames } = this.state
    for(var i = 0; i < cnames.length; i++){
      hold.push(
        <option value={i}>{cnames[i] + ' - ' + cstocks[i]}</option>
      )
    }
    return(
      <select onChange={this.AddTicker} size={10} style={{width:300, height: 75, fontSize:15, backgroundColor: 'yellow', color: 'black'}}>
        {hold}
      </select>
    )
  }

  DisplayTicks(){
    const hold = []
    const { tickers } = this.state
    hold.push(
      <td>Selected Stocks:</td>
    )
    hold.push(
      <td>&nbsp;</td>
    )
    Object.keys(tickers).map((key, idx) => {
      hold.push(
        <td>{tickers[key]}</td>
      )
      hold.push(
        <td>&nbsp;</td>
      )
    })
    return(
      <tr style={{backgroundColor: 'black', color: 'yellow'}}>
        {hold}
      </tr>
    )
  }

  DisplayParams(){
    const { epochs, window, output, lr, prop, ma_vol, skip } = this.state
    const hold = []
    const gold = []
    const wealth = []
    gold.push(<td>Epochs&nbsp;</td>)
    gold.push(<td>Window&nbsp;</td>)
    gold.push(<td>Output&nbsp;</td>)
    gold.push(<td>LearnRate&nbsp;</td>)
    gold.push(<td>Proportion&nbsp;</td>)
    gold.push(<td>LookBack&nbsp;</td>)
    gold.push(<td>Skip&nbsp;</td>)
    hold.push(<td>{epochs}</td>)
    hold.push(<td>{window}</td>)
    hold.push(<td>{output}</td>)
    hold.push(<td>{lr}</td>)
    hold.push(<td>{prop}</td>)
    hold.push(<td>{ma_vol}</td>)
    hold.push(<td>{skip}</td>)
    wealth.push(<td><input name="epochs" value={this.state.epochs} style={{backgroundColor: 'black', color: 'yellow', width: 70, textAlign: 'center'}} onChange={this.adjustInputs}/></td>)
    wealth.push(<td><input name="window" value={this.state.window} style={{backgroundColor: 'black', color: 'yellow', width: 70, textAlign: 'center'}} onChange={this.adjustInputs}/></td>)
    wealth.push(<td><input name="output" value={this.state.output} style={{backgroundColor: 'black', color: 'yellow', width: 70, textAlign: 'center'}} onChange={this.adjustInputs}/></td>)
    wealth.push(<td><input name="lr" value={this.state.lr} style={{backgroundColor: 'black', color: 'yellow', width: 70, textAlign: 'center'}} onChange={this.adjustInputs}/></td>)
    wealth.push(<td><input name="prop" value={this.state.prop} style={{backgroundColor: 'black', color: 'yellow', width: 70, textAlign: 'center'}} onChange={this.adjustInputs}/></td>)
    wealth.push(<td><input name="ma_vol" value={this.state.ma_vol} style={{backgroundColor: 'black', color: 'yellow', width: 70, textAlign: 'center'}} onChange={this.adjustInputs}/></td>)
    wealth.push(<td><input name="skip" value={this.state.skip} style={{backgroundColor: 'black', color: 'yellow', width: 70, textAlign: 'center'}} onChange={this.adjustInputs}/></td>)
    return(
      <Fragment>
        <tr style={{backgroundColor: 'yellow', color: 'black'}}>
          {gold}
        </tr>
        <tr style={{backgroundColor: 'yellow', color: 'black'}}>
          {hold}
        </tr>
        <tr>
          {wealth}
        </tr>
      </Fragment>
    )
  }

  inputChange(evt){
    this.setState({ [evt.target.name]: evt.target.value})
    const { enter, stocks, names } = this.state
    const xnames = []
    const xticks = []
    for(var i = 0; i < stocks.length; i++){
      if(names[i].toLowerCase().includes(enter.toLowerCase())){
        xnames.push(names[i])
        xticks.push(stocks[i])
      }
    }
    this.setState({ cstocks: xticks, cnames: xnames })
  }

  adjustInputs(evt){
    this.setState({[evt.target.name]: evt.target.value })
  }

  fetchData(evt){
    const { tickers, epochs, window, output, lr, prop, ma_vol, skip, sock } = this.state
    const msg = {'tickers': tickers, 'params':[epochs, window, output, lr, prop, ma_vol, skip]}
    sock.send(JSON.stringify(msg))
    evt.preventDefault()
  }

  clearData(evt){
    this.setState({ tickers: {}})
  }

  plotEF(){
    const hold = []
    const { df } = this.state
    if(df !== null){
      const analyzer = []
      analyzer.push(
        {x:df['hist_points']['x'],
        y:df['hist_points']['y'],
        text:df['names'],
        type:'scatter',
        mode:'markers',
        marker:{
        color: 'red'
        },
        name: 'Historical Risk/Return'
      },
      {x:df['hist_sharpe']['x'],
        y:df['hist_sharpe']['y'],
        type:'scatter',
        mode:'markers',
        marker:{
        color: 'white'
        },
        name: 'Historical Sharpe Ratio'
      },
      {x:df['hist_curve']['x'],
        y:df['hist_curve']['y'],
        type:'scatter',
        mode:'lines',
        marker:{
        color:'red'
        },
        name: 'Historical EF Curve'
      },
      {x:df['future_points']['x'],
        y:df['future_points']['y'],
        text:df['names'],
        type:'scatter',
        mode:'markers',
        marker:{
        color:'limegreen'
        },
        name: 'Forecasted Risk/Return'
      },
      {x:df['future_sharpe']['x'],
        y:df['future_sharpe']['y'],
        type:'scatter',
        mode:'markers',
        marker:{
        color: 'white'
        },
        name: 'Forecasted Sharpe Ratio'
      },
      {x:df['future_curve']['x'],
        y:df['future_curve']['y'],
        type:'scatter',
        mode:'lines',
        marker:{
        color:'limegreen'
        },
        name: 'Forecasted EF Curve'
      },
      {
        x:df['hist_line']['x'],
        y:df['hist_line']['y'],
        type:'scatter',
        mode:'lines',
        marker:{
          color: 'red'
        },
        name:'Historical Capital Allocation Line'
      },
      {
        x:df['future_line']['x'],
        y:df['future_line']['y'],
        type:'scatter',
        mode:'lines',
        marker:{
          color: 'limegreen'
        },
        name:'Future Capital Allocation Line'
      }
      )
      df['tickers'].forEach((tick) => {
        analyzer.push(
          {
            x:df['trend'][tick]['x'],
            y:df['trend'][tick]['y'],
            type:'scatter',
            mode:'lines',
            marker: {
              color: 'gray'
            },
            line: {
              dash: 'dot',
              width: 2
            }
          }
        )
      })
      hold.push(
        <Plot
          data={analyzer}
          layout={{
            title: {
              text: 'Optimized Portfolio',
              font: {
                color: 'yellow'
              }
            },
            plot_bgcolor: 'black',
            paper_bgcolor: 'black',
            xaxis:{
              color: 'yellow'
            },
            yaxis:{
              color: 'yellow'
            },
            width: 1100,
            height: 650,
            showlegend: false
          }}
        />
      )
    }
    return hold
  }

  plotPrices(){
    const hold = []
    const { df } = this.state
    if(df !== null){
      df['tickers'].forEach((t) => {
        hold.push(
          <Plot
            data={[{
              x: df['prices'][t]['hist_x'],
              y: df['prices'][t]['hist_y'],
              type:'scatter',
              mode:'lines',
              marker:{
                color:'red'
              }
            },{
              x: df['prices'][t]['pred_x'],
              y: df['prices'][t]['pred_y'],
              type:'scatter',
              mode:'lines',
              marker:{
                color:'limegreen'
              }
            }]}
            layout={{
              title: {
                text: df['svm'][t],
                font:{
                  color: 'yellow'
                }
              },
              xaxis: {
                tickvals: df['prices'][t]['hist_x'],
                ticktext: df['dates'],
                color: 'yellow'
              },
              yaxis: {
                color: 'yellow'
              },
              plot_bgcolor: 'black',
              paper_bgcolor: 'black',
              showlegend: false
            }}
          />
        )
      })
    }

    return hold 
  }

  render(){

    return(
      <Fragment>
        <center>
          {this.titleBar("Portfolio Forecaster")}
          <Tabs>
            <TabList style={{backgroundColor:'yellow'}}>
              <Tab>Home</Tab>
              <Tab>Portfolio</Tab>
              <Tab>Prices</Tab>
            </TabList>
            <TabPanel>
              <br/>
              <input name="enter" value={this.state.enter} onChange={this.inputChange} style={{backgroundColor: 'yellow', color: 'black', textAlign: 'center', width: 300}}/>
              <br/>
              <br/>
              <div>{this.spInputs()}</div>
              <br/>
              <div>{this.DisplayTicks()}</div>
              <br/>
              <div>{this.DisplayParams()}</div>
              <br/>
              <br/>
              <button type="button" onClick={this.fetchData} style={{width: 120, backgroundColor: 'black', color:'yellow'}}>Fetch Data</button>
              <button type="button" onClick={this.clearData} style={{width: 120, backgroundColor: 'black', color:'yellow'}}>Clear</button>
              <br/>
              <br/>
              <div style={{backgroundColor: 'yellow', color: 'black'}}>{this.state.run_state}</div>
              <br/>
            </TabPanel>
            <TabPanel>
              <div>{this.plotEF()}</div>
            </TabPanel>
            <TabPanel>
              <div>{this.plotPrices()}</div>
            </TabPanel>
          </Tabs>
        </center>
      </Fragment>
    )
  }
}

