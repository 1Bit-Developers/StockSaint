# StockSaint
## Visualize | Analyse | Predict

<br/>
StockSaint is a free, online Stock Market based Responsive Web-Application. It allows users to Visualize, Analyse and Predict the Stock Market Movements based on some commonly used Market Dynamics. Our Goal is to generate the best yet simple visualizations for any person who is a Novice or an Advanced in Stock Analysis.

<br/> <br/>
<img src="https://github.com/1Bit-Developers/StockSaint/blob/main/public/site_usage_ss/readme_fast.gif">
<br/> <br/>

## How it Works?
### 1) FrontEnd 

### 2) BackEnd - NodeJS

### 3) BackEnd - Graph Plotting with Python
<ul type='circle'>
  <li>
    Packages used:
    <ol type="1">
      <li><strong>Numpy</strong> - For Data Analysis</li>
      <li><strong>Pandas</strong> - For Data Analysis and cleaning</li>
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
      <li>Matploblib plots the Original Graph and the Trend Graph.</li>
      <li>Slope of the graph is used to calculate the prediction and Market sentiment.</li>
      <li>Caluculate results are returned for display.</li>
    </ol>
  </li>
</ul>
