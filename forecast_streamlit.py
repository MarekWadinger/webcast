# streamlit run forecast_streamlit.py
import streamlit as st
import pandas as pd
import base64
import json
import time

from prophet.serialize import model_to_json
from prophet.plot import plot_plotly, plot_components_plotly

from loadforecast.forecaster import LoadProphet
from loadforecast.model_load import model_load
from src.anomaly import resample_data, anomaly_rate


@st.cache_data()
def modify_data(csv_data_file):
    df = pd.read_csv(csv_data_file)
    df_new = df[['DateTime', 'Load']].rename(columns={'DateTime': 'ds', 'Load': 'y'})
    df_new['ds'] = pd.to_datetime(df_new['ds'])
    return df_new


@st.cache_resource()
def build_model(df, pre_model, country, chps, sps, hps, ds, ws, ys):
    model = LoadProphet(df, pretrained_model=pre_model, country=country, changepoint_prior_scale=chps,
                        seasonality_prior_scale=sps, holidays_prior_scale=hps, daily_seasonality=ds,
                        weekly_seasonality=ws, yearly_seasonality=ys)
    return model


@st.cache_resource()
def predict_future(_model, period):
    return _model.prediction(prediction_periods=period*24*4, frequency='15min')


def save_csv(df_forecast):
    csv = df_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv().encode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="load_forecast.csv" target="_blank">Download csv file</a>'
    return href


def save_json(df_forecast):
    json_file = df_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_json().encode()
    b64 = base64.b64encode(json_file).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="load_forecast.json" target="_blank">Download json file</a>'
    return href


def model_json(model):
    json_file = model_to_json(model)
    json_file = json.dumps(json_file)
    b64 = base64.b64encode(json_file.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="loadForecast_model.json" ' \
           f'target="_blank">Download model</a>'
    return href


# Build web app
# -----------------------------------------------------------------------------
st.title("""Electrical load forecast 
Using  *fbprophet*""")
col1, col2 = st.columns(2)
with col1:
    data_file = st.file_uploader('Import data as csv file. Data should have two columns named DateTime and Load.')
with col2:
    model_file = st.file_uploader('Import pretrained model as json file. Without pretrained model app trains a new one.')
user_period = st.slider('Prediction period in days:', min_value=1, max_value=14, value=1, step=1, )
user_plot = st.button('Plot forecast')
p1 = st.empty()
p2 = st.empty()
p3 = st.empty()

# Build Sidebar
with st.sidebar:
    st.title('Customize model')
    user_country = st.text_input('Insert ISO code of Country', value='BE')
    st.text('Take local holidays into account when training a model.')
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
    data_frame = modify_data(data_file)

    if model_file:
        attr_dict = json.loads(json.load(model_file))
        m = model_load(attr_dict)
        df_resampled, frequency = resample_data(data_frame)
        anomaly_proportion = anomaly_rate(m, df_resampled, frequency, plot=False)
        p1.info('Anomaly proportion between prediction and validation data is **%.2f %%**' % (anomaly_proportion*100))
        if anomaly_proportion > 0.1:
            p2.info('Model is outdated. Warm fitting new model...')
            start_time = time.time()
            m = build_model(data_frame, m, user_country, user_chps, user_sps, user_hps, user_ds, user_ws, user_ys)
            fitting_time = time.time() - start_time
        else:
            p2.info('Model is up-to-date and ready for prediction.')
            fitting_time = 0
    else:
        p2.info('Fitting new model...')
        start_time = time.time()
        m = build_model(data_frame, None, user_country, user_chps, user_sps, user_hps, user_ds, user_ws, user_ys)
        fitting_time = time.time() - start_time

    p3.info('Making prediction...')
    start_time = time.time()
    forecast = predict_future(m, user_period)
    prediction_time = time.time() - start_time
    p3.success('Prediction made with success. Fitting time: **%.2f s**. Prediction time: **%.2f s**.' % (fitting_time, prediction_time))

    st.subheader('Download prediction')
    st.markdown(save_json(forecast), unsafe_allow_html=True)
    st.markdown(save_csv(forecast), unsafe_allow_html=True)
    st.markdown(model_json(m), unsafe_allow_html=True)

if user_plot:
    st.plotly_chart(plot_plotly(m, forecast))
    st.plotly_chart(plot_components_plotly(m, forecast))

time.sleep(1.2)
p1.empty()
time.sleep(0.95)
p2.empty()
