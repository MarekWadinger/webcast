# Web App for Load Forecasting

https://share.streamlit.io/marekwadinger/semestral_project/forecast_streamlit.py

This project is integration of [loadforecast](https://github.com/MarekWadinger/loadforecast) an automatic time 
series forecasting procedure of an electrical load based on 
[prophet](https://github.com/facebook/prophet) and [pyod](https://github.com/yzhao062/pyod) anomaly detection toolkit 
into user-friendly web app environment of [streamlit](https://github.com/streamlit/streamlit). 

It serves for automation of whole procedure of electrical load forecasting, making it fast and easy to use for both 
research workers and users with limited programming skills.

## ⚡️ Quickstart

Quick way to get your hand on the load forecasting service is to use our [Streamlit app](https://share.streamlit.io/marekwadinger/semestral_project/forecast_streamlit.py) to play around with example data.

As a quick example, we will run the online service use historical load data to fit new model and get prediction
1. To start the service, open [Streamlit app](https://share.streamlit.io/marekwadinger/semestral_project/forecast_streamlit.py)
2. Import the [historical load data](https://github.com/MarekWadinger/webcast/blob/master/data/load_20220513-20220612.csv)
3. The service silently fits your model
4. Choose prediction period to get your prediction
5. Visualize it pressing "Plot forecast"
6. Download the prediction as JSON pressing "Download json file" or CSV  pressing "Download csv file"
7. Download the model pressing "Download model" for later use

Note: If the app is sleeping, you can build it by pressing the big red button on screen.

Now we'd like to make prediction while new historical data are available. We have our previously fitted model at our hands
1. Import the [extended historical load data](https://github.com/MarekWadinger/webcast/blob/master/data/load_20220513-20220620.csv)
2. Import the [model](https://github.com/MarekWadinger/webcast/blob/master/model/loadForecast_model.json)
3. The service silently checks whether the model is outdated and refits it when needed
4. Get your prediction
5. Download the updated model pressing "Download model"

## ✏️ Customize the model

The side panel allows you to customize your prediction model:
* Includes local holidays using coutry's ISO code
* Offers rich control over various time-series attributes

 ## 👐 Contributing

Feel free to contribute in any way you like, we're always open to new ideas and approaches.

- Feel welcome to [open an issue](https://github.com/MarekWadinger/webcast/issues/new/choose) if you think you've spotted a bug or a performance issue.


<!-- 
## 📝 License

This algorithm is free and open-source software licensed under the .
  -->
