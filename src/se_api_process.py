import pandas as pd
import json

def get_se_as_df(filename):
    with open(filename) as f:
        data = json.loads(f.read())
    for record in data:
        for key, value in record.items():
            if type(value)==dict:
                # extract only kWh
                kWh = value['energy_kWh']
                record[key] = kWh
    df = pd.DataFrame(data)
    # convert string to datetime object
    df['created_on'] = pd.to_datetime(df['created_on'])
    df = df.set_index('created_on')
    return df

if __name__=='__main__':
    filename = '../se_daily.json'
    df = get_se_as_df(filename)
    print(df.head)
