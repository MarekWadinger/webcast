#streamlit run C:\Users\marek\PycharmProjects\pythonProject\Load_forecast_streamlit.py
import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import base64


@st.cache(suppress_st_warning=True)
def prediction(m, prediction_period):
    # Extend df to the future by specified number of hours
    future = m.make_future_dataframe(periods=prediction_period * 24 * 4 , freq='15min')
    # Make prediction
    forecast = m.predict(future)

    return m, forecast

@st.cache(suppress_st_warning=True)
def build_model(data_file, country, chps, sps , hps):
    # Import the csv file as pandas data frame
    df = pd.read_csv(data_file)
    df_new = df[['DateTime', 'Load']].rename(columns={"DateTime": "ds", "Load": "y"})
    
    m = Prophet(changepoint_prior_scale = chps, seasonality_prior_scale = sps, holidays_prior_scale = hps,
            daily_seasonality = True, weekly_seasonality=28, yearly_seasonality = False)
    # Add local holidays
    m.add_country_holidays(country)
    # Call fit method with historical data
    m.fit(df_new)
    
    return m
    
def save_csv(forecast):
    csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv().encode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="load_forecast.csv" target="_blank">Download csv file</a>'
    
    return href

def save_json(forecast):
    json = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_json().encode()
    b64 = base64.b64encode(json).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="load_forecast.json" target="_blank">Download json file</a>'

    return href

# Build web app
# -----------------------------------------------------------------------------
st.write("""# Electrical load forecast
Using  *fbprophet*""")
file = st.file_uploader("Import csv file")
# user_date = st.date_input("Date")
# user_date = user_date.strftime("%Y-%m-%d")
# st.text(user_date)
user_periods = st.slider("Prediction period in days:", 1, 7, 1)
user_plot = st.button("Plot forecast")

# Build Sidebar
st.sidebar.header('Customize model')
user_country = st.sidebar.text_input('Insert ISO code of Country',value="BE")
st.sidebar.text("List of available ISO codes here:")
st.sidebar.markdown("https://github.com/dr-prodigy/python-holidays")
st.sidebar.subheader("Change with caution:")
user_chps = st.sidebar.number_input('Changepoint Prior Scale',min_value=0.000,max_value=100.000,value=0.001)
user_sps = st.sidebar.number_input('Seasonality Prior Scale',min_value=0.00,max_value=100.00,value=0.1)
user_hps = st.sidebar.number_input('Holidays Prior Scale',min_value=0.00,max_value=100.00,value=0.1)
user_ds = st.sidebar.selectbox('Daily Seasonality', ['True','False',"Auto"],index=0)
weekly = st.sidebar.selectbox('Weekly Seasonality', ['True','False',"Auto","Custom"],index=3)
weekly_placeholder = st.sidebar.empty()
user_ys = st.sidebar.selectbox('Yearly Seasonality', ['True','False',"Auto"],index=1)
weekly_placeholder.number_input("Custom weekly seasonality", min_value=0,max_value=100,value=28) if weekly == "Custom" else ""

if file:
    # Import the csv file as pandas data frame
    m = build_model(file,user_country, user_chps, user_sps , user_hps)
    m, forecast = prediction(m, user_periods)
    
    st.subheader('Download prediction')
    st.markdown(save_json(forecast), unsafe_allow_html=True)   
    st.markdown(save_csv(forecast), unsafe_allow_html=True)
    
    if user_plot:
        st.plotly_chart(plot_plotly(m, forecast))
        st.plotly_chart(plot_components_plotly(m, forecast))
