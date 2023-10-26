import time

import pandas as pd
from loadforecast.forecaster import LoadProphet
from matplotlib import pyplot as plt
from pyod.models.iforest import IForest
from sklearn.preprocessing import MinMaxScaler


def get_sampling_frequency(s, ratio):
    """Get sampling frequency of time-series data.

    Function that returns the sampling frequency of time-series data

        Function

        Parameters
        ----------
        ratio:

        s:

        Returns
        -------
        The fitted Prophet object.

        """
    tf = s['ds'].dropna().diff().value_counts()
    counter = 0

    for i in range(len(tf)):
        counter += tf[i]
        if counter / sum(tf[:]) >= ratio:
            break
    if tf.index[i].days == 0:
        freq = str(tf.index[i].seconds) + "S"
    else:
        freq = str(tf.index[i].days) + "D"
    return freq


def resample_data(s: pd.DataFrame, ratio=0.9):
    """Resample time-series data.

    Function

    Parameters
    ----------
    ratio:

    s:

    Returns
    -------
    The fitted Prophet object.

    """

    freq = get_sampling_frequency(s, ratio)

    s = s.set_index('ds').resample(freq).first().reset_index()
    s = s.dropna()

    return s, freq


def anomaly_rate(
        model: LoadProphet,
        validation_df: pd.DataFrame,
        freq,
        plot=False):
    if freq[:-1].isnumeric() and (freq[-1] == 'S' or freq[-1] == 'D'):
        last_history = (model.start + model.t_scale).round(freq)
    else:
        raise ValueError("Unsupported frequency format. "
                         "Provide any valid frequency for pd.date_range, as multiple of 'D' or 'S'.")

    first_validation = validation_df['ds'].iloc[0]
    last_validation = validation_df['ds'].iloc[-1]

    if last_validation > last_history:
        if first_validation <= last_history:
            validation_df = validation_df.loc[validation_df['ds'] > last_history].dropna()[['ds', 'y']]

        start_timer = time.time()
        future = validation_df['ds'].to_frame(name='ds')
        prediction_data = model.predict(future)[['ds', 'yhat']]  # TOO SLOW!
        print("--- Prediction: %s seconds ---" % (time.time() - start_timer))

        df = pd.DataFrame({'y': validation_df['y'].values, 'yhat': prediction_data['yhat'].values})
        scaler = MinMaxScaler(feature_range=(0, 1))
        df[['y', 'yhat']] = scaler.fit_transform(df[['y', 'yhat']])

        clf_name = 'iForest'
        clf = IForest()
        clf.fit(df)

        # get the prediction labels and outlier scores of the training data
        y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
        y_train_scores = clf.decision_scores_  # raw outlier scores

        if plot:
            # fig = plt.figure(facecolor='w', figsize=(10, 6))
            # ax = fig.add_subplot(111)
            # ax.plot(prediction_data['ds'].dt.to_pydatetime(), deviation, 'k.')
            # ax.plot(prediction_data['ds'][y_train_pred == 1].dt.to_pydatetime(), deviation[y_train_pred == 1], 'r.')
            # fig.show()

            fig1 = plt.figure(facecolor='w', figsize=(10, 6))
            ax = fig1.add_subplot(111)
            ax.plot(prediction_data['ds'].dt.to_pydatetime(), y_train_scores)
            ax.plot(prediction_data['ds'][y_train_pred == 1].dt.to_pydatetime(), y_train_scores[y_train_pred == 1], 'r.')
            fig1.show()

            fig2 = plt.figure(facecolor='w', figsize=(10, 6))
            ax = fig2.add_subplot(111)
            ax.plot(validation_df['ds'].dt.to_pydatetime(), validation_df['y'].values)
            ax.plot(prediction_data['ds'].dt.to_pydatetime(), prediction_data['yhat'].values)
            ax.vlines(prediction_data['ds'][y_train_pred == 1].dt.to_pydatetime(), min(validation_df['y'].values), max(validation_df['y'].values), 'r')
            fig2.show()

        return sum(y_train_pred) / len(y_train_pred)

    else:
        raise ValueError("Validation dataset has no data point after last member of time-series of historical data that",
                         "the model was trained on. Please use validation dataset with last member of the time series",
                         "after %s." % last_history)
