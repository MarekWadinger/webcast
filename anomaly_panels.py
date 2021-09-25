from src import se_api_process
from src.anomaly_report import tidy_individual_detection

data = se_api_process.get_se_as_df('se_daily.json')
data = data.drop(data.iloc[26].name)

df_modules, yield_modules, df_outliers, df_scores = tidy_individual_detection(data, plot=True, report=True)
