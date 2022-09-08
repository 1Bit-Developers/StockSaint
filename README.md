# StockSaint
## Visualize | Analyse | Predict

## Endorsed by
### <b>Jugal Agrawal</b>

_Project Manager|  FX & MM Operations Technology    
NatWest Markets CDIO – Core Trading Solutions_

## Why?

We have often looked for a way to ease up our way in stock market predictions but its very hard to predict stock behaviour solely by market patterns especially for beginners ..so we thought of using our historic stock data to our advantage and predict the future trends for us.

## Current Working Web Application Link
https://stocksaint.herokuapp.com/

## How it Works?

### 1) FrontEnd 
* BOOTSTRAP 5.0 used for building the frontend.
* Used different custom CSS templates with modifications.
* Connected the front-end to hyper for sending requests.
* Used Javascript for sliding animations and smooth scrolling.
* Added different API's for the functioning of backend

### 2) BackEnd - NodeJS
* _Express_ Framework used.
* Stock Data taken from _RESTFUL API_ of AlphaVantage.
* Challange was to connect Python scripts to work with NodeJS.
* Used Child Processes to connect with Python.
* Used _Embedded Javascript Templating_ to parse User requested data to Front-End. 

### 3) BackEnd - Graph Plotting with Python
<ul type='circle'>
  <li>
    Packages used:
    <ol type="1">
      <li><strong>Numpy</strong> - For Data Analysis</li>
      <li><strong>Pandas</strong> - For Data Refining and cleaning</li>
      <li><strong>Matplotlib</strong> - For creating plots</li>
      <li><strong>Mplcyberpunk</strong> - For creating "Cyberpunk" Style plots</li>
    </ol>
  </li>
  
  <br>
  
  <li>
    Process:
    <ol type="1">  
      <li>API Data is recieved in CSV format.</li>
      <li>Pandas is used to create and clean the dataframe formed using the data.</li>
      <li>Calculation for the Running Average or Exponential Smoothing is done on basis of window specified.</li>
      <li>Matplotlib plots the Original Graph and the Trend Graph.</li>
      <li>The plots are generated in png format and are saved in the public/tmp folder of the Heroku/Local server.</li>
      <li>Slope of the graph is used to calculate the prediction and Market sentiment.</li>
      <li>Caluculate results are returned for display.</li>
    </ol>
  </li>
</ul>

<br/> <br/>
<img src="https://github.com/1Bit-Developers/StockSaint/blob/main/public/site_usage_ss/readme_fast.gif">
<br/> <br/>
