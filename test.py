from src import se_api_process
from src.anomaly_report import anomaly_report

data = se_api_process.get_se_as_df('src/se_daily.json')

anomaly, reporting = anomaly_report(data)
