import requests
import datetime
import pytz
import json

URL = 'https://tbp-ems-api.deepcontrol.sk/v1/se_telemetry/1c55e09415fe6f40b03c7e8ab3105b02fccc6ef2'
BASE_TIME = datetime.time(hour=23, minute=59)  # ku ktoremu casu nacitat denne data (v lokalnej casovej zone)
FIRST_DATE = datetime.date(year=2021, month=6, day=17)
TODAY_DATE = datetime.date.today()

def combine_date_time(date, time):
    return datetime.datetime.combine(date, time)

def to_utc(dt):
    return dt.astimezone(pytz.utc)

def get_se_daily(url=''):
    daterange = [FIRST_DATE + datetime.timedelta(days=x) for x in range(0, (TODAY_DATE-FIRST_DATE).days)]
    results = []
    for d in daterange:
        before = to_utc(combine_date_time(d, BASE_TIME))
        print(before)
        r = requests.get(URL, {'before': before.isoformat()})
        data = json.loads(r.text.replace('Panel', 'Module'))
        results.append(data)
    return results

if __name__=='__main__':
    se = get_se_daily()
    with open('../se_daily.json', 'w') as f:
        f.write(json.dumps(se))
    print('Data written to "se_daily.json"')
