# streamlit run C:\Users\marek\PycharmProjects\pythonProject\Load_forecast_streamlit.py
import streamlit as st
import pandas as pd
import base64
import json

from prophet.serialize import model_to_json
from prophet.plot import plot_plotly, plot_components_plotly

from loadforecast.forecaster import LoadProphet
from loadforecast.model_load import model_load


@st.cache(suppress_st_warning=True)
def prediction(model, prediction_period):
    # Make prediction on extend df to the future by specified number of hours
    forecast = model.prediction(prediction_periods=prediction_period * 24 * 4, frequency='15min')
    return forecast


@st.cache(suppress_st_warning=True)
def build_model(data_file, pre_model, country, chps, sps, hps, ds, ws, ys):
    # Import the csv file as pandas data frame
    df = pd.read_csv(data_file)
    df_new = df[['DateTime', 'Load']].rename(columns={'DateTime': 'ds', 'Load': 'y'})
    df_new['ds'] = pd.to_datetime(df_new['ds'])
    model = LoadProphet(df_new, pretrained_model=pre_model, country=country, changepoint_prior_scale=chps,
                        seasonality_prior_scale=sps, holidays_prior_scale=hps, daily_seasonality=ds,
                        weekly_seasonality=ws, yearly_seasonality=ys)
    return model


def save_csv(forecast):
    csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv().encode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="load_forecast.csv" target="_blank">Download csv file</a>'
    return href


def save_json(forecast):
    json_file = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_json().encode()
    b64 = base64.b64encode(json_file).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="load_forecast.json" target="_blank">Download json file</a>'
    return href


def model_json(model):
    json_file = model_to_json(model)
    json_file = json.dumps(json_file)
    b64 = base64.b64encode(json_file.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="loadForecast_model.json" ' \
           f'target="_blank">Download model json file</a>'
    return href


# Build web app
# -----------------------------------------------------------------------------
st.write("""# Electrical load forecast 
Using  *fbprophet*""")
col1, col2 = st.beta_columns(2)
with col1:
    data_file = st.file_uploader('Import data as csv file')
with col2:
    model_file = st.file_uploader('Import pretrained model as json file')
user_periods = st.slider('Prediction period in days:', min_value=1, max_value=7, value=1, step=1)
user_plot = st.button('Plot forecast')

# Build Sidebar
with st.sidebar:
    st.header('Customize model')
    user_country = st.text_input('Insert ISO code of Country', value='BE')
    st.caption('List of available ISO codes here:')
    st.caption('https://github.com/dr-prodigy/python-holidays')
    st.subheader('Change with caution:')
    with st.form(key='Model Parameters'):
        user_chps = st.slider('Changepoint Prior Scale', min_value=0.001, max_value=0.050, value=0.001, step=0.001, format='%.3f')
        user_sps = st.slider('Seasonality Prior Scale', min_value=0.01, max_value=10.00, value=10.00, step=0.01)
        user_hps = st.slider('Holidays Prior Scale', min_value=0.01, max_value=10.00, value=0.1, step=0.01)
        user_ds = st.selectbox('Daily Seasonality', [True, False, 'auto'], index=0)
        user_ws = st.selectbox('Weekly Seasonality', [True, False, 'auto', 'Custom'], index=3)
        st.caption('To use "Custom" Weekly Seasonality submit "Custom" and then change value')
        weekly_placeholder = st.empty()
        user_ys = st.selectbox('Yearly Seasonality', [True, False, 'auto'], index=1)
        submit_button = st.form_submit_button(label='Submit')
        if user_ws == 'Custom':
            user_ws = weekly_placeholder.slider('Weekly seasonality', min_value=0, max_value=100, value=28, step=2)


if data_file:
    if model_file:
        attr_dict = json.loads(json.load(model_file))
        m1 = model_load(attr_dict)
    else:
        m1 = None
    # Import the csv file as pandas data frame
    m = build_model(data_file, m1, user_country, user_chps, user_sps, user_hps, user_ds, user_ws, user_ys)
    forecast = prediction(m, user_periods)
    
    st.subheader('Download prediction')
    st.markdown(save_json(forecast), unsafe_allow_html=True)   
    st.markdown(save_csv(forecast), unsafe_allow_html=True)
    st.markdown(model_json(m), unsafe_allow_html=True)

    if user_plot:
        st.plotly_chart(plot_plotly(m, forecast))
        st.plotly_chart(plot_components_plotly(m, forecast))
