#streamlit run C:\Users\marek\PycharmProjects\pythonProject\Load_forecast_streamlit.py
import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import base64


@st.cache(suppress_st_warning=True)
def prediction(data_file, prediction_period):
    # Import the csv file as pandas data frame
    df = pd.read_csv(data_file)
    df_new = df[['DateTime', 'Load']].rename(columns={"DateTime": "ds", "Load": "y"})
    m = Prophet(changepoint_prior_scale=0.001,seasonality_prior_scale=0.1,holidays_prior_scale=0.1,
            daily_seasonality=True,weekly_seasonality=28,yearly_seasonality=False)
    # Add local holidays
    m.add_country_holidays('BE')
    # Call fit method with historical data
    m.fit(df_new)
    # Extend df to the future by specified number of hours
    future = m.make_future_dataframe(periods=prediction_period * 24 * 4 , freq='15min')
    # Make prediction
    forecast = m.predict(future)

    
    return m, forecast

# Build web app
# -----------------------------------------------------------------------------
st.write("""# Electrical load forecast
Using  *fbprophet*""")
file = st.file_uploader("Pick a file")
# user_date = st.date_input("Date")
# user_date = user_date.strftime("%Y-%m-%d")
# st.text(user_date)
user_periods = st.slider("Prediction period in days:", 1, 7, 1)
#user_plot = st.button('Plot forecast')
user_save_json = st.button('Save as json')
user_save_csv = st.button('Save as csv')

if file:
    # Import the csv file as pandas data frame
    m, forecast = prediction(file, user_periods)
    #if user_plot:
    
    if user_save_json:
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_json(str(user_periods) +'_day_load_forecast.json')
    if user_save_csv:
        csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(str(user_periods) +'_day_load_forecast.csv').encode()
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="captura.csv" target="_blank">Download csv file</a>'
        st.markdown(href, unsafe_allow_html=True)
    st.plotly_chart(plot_plotly(m, forecast))
    st.plotly_chart(plot_components_plotly(m, forecast))
        
        
