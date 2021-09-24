import numpy as np
import pandas as pd

from typing import Union
from pandas.core.frame import DataFrame
from scipy.stats import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pyod.models.copod import COPOD
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM

from src.se_api_download import get_se_daily
from src.se_api_process import get_se_as_df

import matplotlib
from matplotlib import pyplot as plt
import plotly.express as px
import plotly
plotly.offline.init_notebook_mode(connected=True)


class Report:
    pass


def drop_missing(df: pd.DataFrame, valid_col_rate: float = 0.1) -> pd.DataFrame:
    """ Drop invalid columns and rows with missing values.

    Parameters
        ----------
        df: pd.DataFrame

        valid_col_rate: float representing the maximum % proportion of NAs in column
                        to be still considered valid. 0 means that we drop all columns
                        having at last one missing.

    Returns
        -------
        pd.DataFrame with invalid columns and NA entries dropped from it.

    """
    if valid_col_rate >= 1:
        valid_col_rate = valid_col_rate / 100

    sum_na = df.isna().sum()
    cols_drop = sum_na[sum_na / df.shape[0] > valid_col_rate].index.values
    df = df.drop(columns=cols_drop)
    df = df.dropna()

    print('Dropped {} columns: {}'.format(cols_drop.shape[0], cols_drop))
    return df


def get_yield(df: pd.DataFrame, name_subset: str, name_set: str) -> pd.DataFrame:
    """ Return product of row-wise division.

    Perform row-wise division between transposed subset and set to which it belongs.

    Parameters
        ----------
        df: pd.DataFrame with column names containing: '.*[name_subset]\..*'
            and '.*[name_set]\..*'.

        name_subset: string specific for subset columns name. Sum of subset cols for
                     one measurement should be equal to set.

        name_set: string specific for set column name.

    Returns
        -------
        pd.DataFrame with product of row-wise division with the same shape as df_subset.

    """

    df_subset = df[[i for i in df if name_subset in i]]
    df_set = df[[i for i in df if name_set in i]]

    subsets_yield = []
    # Works with arrays
    for set_id in df_set:
        set_num = set_id.split(' ')[1] + '.'
        # Returns array of modules belonging to given string
        subsets = df_subset[[module for module in df_subset if set_num in module]].T.values
        temp_yield = np.divide(subsets, df_set[set_id].values)
        subsets_yield = [*subsets_yield, *temp_yield]
    # Transforms array to DataFrame
    subsets_yield = pd.DataFrame(np.transpose(subsets_yield), index=df_subset.index, columns=df_subset.columns)

    return subsets_yield


def plot_outlier_detection(df_floats: pd.DataFrame, y_pred: np.ndarray, clf, clf_name: str = None, scaler=None):
    """ Return contourf plot of the anomaly detection model.

        Plots contourf plot of decision scores marking area where datapoints would be considered inliers.

        Parameters
            ----------
            df_floats: pd.DataFrame with elements as floats.

            y_pred: numpy array of the same length as df_floats that assigns 0/1 (inlier/outlier) to each observation
                    according to fitted model.

            clf: fitted model.

            clf_name: name of fitted model.

            scaler: estimator that was used to change range of df_floats DataFrame

        Returns
            -------
            Contourf plot.

        """
    if df_floats.shape[1] > 2:
        print('Plotting first two variables')
    elif df_floats.shape[1] < 2:
        print('Sorry can not plot less than two variables')

    # predict raw anomaly score
    scores_pred = clf.decision_function(df_floats.iloc[:, [0, 1]]) * -1
    # threshold value to consider a datapoint inlier or outlier
    threshold = stats.scoreatpercentile(scores_pred, 100 * len(y_pred[y_pred == 1]) / len(y_pred))

    # Specifies interval over which the np.linspace will be created
    x_lim_min, x_lim_max = df_floats.iloc[:, 0].min(), df_floats.iloc[:, 0].max()
    x_delta = 0.05 * (x_lim_max - x_lim_min)
    y_lim_min, y_lim_max = df_floats.iloc[:, 1].min(), df_floats.iloc[:, 1].max()
    y_delta = 0.05 * (y_lim_max - y_lim_min)
    # coordinate array for vectorized evaluation of raw anomaly scores
    # TODO: Coarser grid sometimes returns error when plotting (threshold out of [zz.min();zz.max()]).
    xx, yy = np.meshgrid(np.linspace(x_lim_min - x_delta, x_lim_max + x_delta, 100),
                         np.linspace(y_lim_min - y_delta, y_lim_max + y_delta, 100))
    # decision function calculates the raw anomaly score for every point
    zz = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
    zz = zz.reshape(xx.shape)

    # undo the scaling so the plot is in the same scale as input data
    if scaler:
        df_floats.iloc[:, [0, 1]] = scaler.inverse_transform(df_floats.iloc[:, [0, 1]])
        x_lim_min, x_lim_max = df_floats.iloc[:, 0].min(), df_floats.iloc[:, 0].max()
        x_delta = 0.05 * (x_lim_max - x_lim_min)
        y_lim_min, y_lim_max = df_floats.iloc[:, 1].min(), df_floats.iloc[:, 1].max()
        y_delta = 0.05 * (y_lim_max - y_lim_min)
        xx, yy = np.meshgrid(np.linspace(x_lim_min - x_delta, x_lim_max + x_delta, 100),
                             np.linspace(y_lim_min - y_delta, y_lim_max + y_delta, 100))

    # inliers_1 - inlier feature 1,  inliers_2 - inlier feature 2
    inliers_1 = (df_floats.iloc[:, 0][y_pred == 0]).values.reshape(-1, 1)
    inliers_2 = (df_floats.iloc[:, 1][y_pred == 0]).values.reshape(-1, 1)
    # outliers_1 - outlier feature 1, outliers_2 - outlier feature 2
    outliers_1 = df_floats.iloc[:, 0][y_pred == 1].values.reshape(-1, 1)
    outliers_2 = df_floats.iloc[:, 1][y_pred == 1].values.reshape(-1, 1)

    plt.figure(figsize=(10, 10))
    # fill blue map colormap from minimum anomaly score to threshold value
    plt.contourf(xx, yy, zz, levels=np.linspace(zz.min(), threshold, 50), cmap=plt.cm.Blues_r)
    plt.colorbar(plt.contourf(xx, yy, zz, levels=np.linspace(zz.min(), threshold, 50), cmap=plt.cm.Blues_r))
    # fill orange contour lines where range of anomaly score is from threshold to maximum anomaly score
    plt.contourf(xx, yy, zz, levels=[threshold, zz.max()], colors='orange')
    # draw red contour line where anomaly score is equal to threshold
    a = plt.contour(xx, yy, zz, levels=[threshold], linewidths=2, colors='red')
    # draw inliers as white dots
    b = plt.scatter(inliers_1, inliers_2, c='white', s=20, edgecolor='k')
    # draw outliers as black dots
    c = plt.scatter(outliers_1, outliers_2, c='black', s=20, edgecolor='k')

    plt.axis('tight')
    # loc=2 is used for the top left corner
    plt.legend(
        [a.collections[0], b, c],
        ['learned decision function', 'inliers', 'outliers'],
        prop=matplotlib.font_manager.FontProperties(size=20),
        loc=2)

    plt.xlim((x_lim_min - x_delta, x_lim_max + x_delta))
    plt.ylim((y_lim_min - y_delta, y_lim_max + y_delta))
    plt.xlabel(df_floats.columns[0])
    plt.ylabel(df_floats.columns[1])
    if clf_name:
        plt.title(clf_name)
    plt.show()

    return


def detect_anomaly(df_floats: pd.DataFrame, train_size: float, outliers_rate: float, classifier: str,
                   plot: bool = False):
    """ Return binary classified outlier and raw outlier score.

    Performs training of anomaly detection model on subset of dataset and returns
    binary label and decision score for whole dataset.

    Parameters
        ----------
        df_floats: pd.DataFrame with elements as floats.

        train_size: proportion of dataset to be used for training anomaly detection model.

        outliers_rate: proportion of training set to be considered outlier.

        classifier: string representing name of anomaly detection algorithm.

        plot: plots 2d contourf of anomaly detection scores.

    Returns
        -------
        y_pred: numpy array of the same length as df_floats that assigns 0/1 (inlier/outlier) to each observation
                    according to fitted model.
        y_scores: numpy array of the same length as df_floats that assigns outlier scores to each observation
                    according to fitted model.

    """
    if df_floats.shape[0] < 8:
        raise Warning('Not enough measurements. Please use DataFrame with at last 10 measurements.')
    if train_size > 1:
        train_size = train_size / 100
    # TODO: Find out empirical way to set contamination level - Tukey's method
    if outliers_rate >= 1:
        outliers_rate = outliers_rate / 100

    random_state = np.random.RandomState(42)

    # TODO: Perform scaling of data ONLY for AKNN, CBLOF, HBOS, KNN, OCSVM. Other classifiers are not influenced.
    classifiers = {
        'Average KNN (AKNN)': KNN(method='mean', contamination=outliers_rate),
        'Cluster-based Local Outlier Factor (CBLOF)': CBLOF(contamination=outliers_rate, check_estimator=False,
                                                            random_state=random_state),
        'Copula based Outlier Detection (COPOD)': COPOD(contamination=outliers_rate),
        'Histogram-base Outlier Detection (HBOS)': HBOS(contamination=outliers_rate),
        'Isolation Forest (IForest)': IForest(contamination=outliers_rate, random_state=random_state),
        'K Nearest Neighbors (KNN)': KNN(contamination=outliers_rate),
        'One-Class SVM (OCSVM)': OCSVM(contamination=outliers_rate),
        'Principal component analysis (PCA)': PCA(contamination=outliers_rate)
    }

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df_floats.iloc[:, [0, 1]])
    df_scaled = pd.DataFrame(scaled, index=df_floats.index, columns=df_floats.columns)

    x_train, x_test = train_test_split(df_scaled, train_size=train_size)

    if classifier == 'all':
        raise Warning('This option is currently unsupported.'
                      '\nPlease use one of those classifiers:'
                      '\n{}.'.format(list(classifiers.keys())))
        # for i, (clf_name, clf) in enumerate(classifiers.items()):
        #    # fit model
        #    clf.fit(x_train)
        #    # prediction of a datapoint category outlier or inlier
        #    y_pred = clf.predict(df_scaled)
        #    plot_outlier_detection(df_scaled, y_pred, clf, clf_name, scaler)
    else:
        clf_name = ''
        for name in classifiers.keys():
            if classifier in name:
                clf_name = name
                break
        if clf_name:
            print("\nUsed classifier: {}".format(clf_name))
            clf = classifiers.get(clf_name)
            clf.fit(x_train)
            y_pred = clf.predict(df_scaled)  # binary labels (0: inliers, 1: outliers)
            y_scores = clf.decision_function(df_scaled)  # raw outlier scores
        else:
            raise NameError('Unknown classifier. '
                            'Please use one of those: {}.'.format(list(classifiers.keys())))

        if plot:
            plot_outlier_detection(df_scaled, y_pred, clf, clf_name, scaler)

    return y_pred, y_scores


def report_anomaly(df_long):
    df_long_anomaly = df_long[df_long['outlier'] == 1]

    if df_long_anomaly.shape[0] > 0:
        report_obj = Report()
        report_obj.anomaly_rate = round(df_long_anomaly.shape[0] / df_long.shape[0] * 100, 2)

        report_obj.bad_module = df_long_anomaly.groupby(['variable']).count()['outlier'].idxmax()
        sum_bad_module_anomaly = df_long_anomaly[df_long_anomaly['variable'] == report_obj.bad_module].shape[0]
        sum_bad_module = df_long[df_long['variable'] == report_obj.bad_module].shape[0]
        report_obj.bad_module_rate = round(sum_bad_module_anomaly / sum_bad_module * 100, 2)

        report_obj.bad_day = df_long_anomaly.groupby(['created_on']).count()['outlier'].idxmax()
        sum_bad_day_anomaly = df_long_anomaly[df_long_anomaly['created_on'] == report_obj.bad_day].shape[0]
        sum_bad_day = df_long[df_long['created_on'] == report_obj.bad_day].shape[0]
        report_obj.bad_day_rate = round(sum_bad_day_anomaly / sum_bad_day * 100, 2)

        print("""\nREPORT\
                 \n--------------------\n{} requires your attention!\
                 \n{:.2f}% of observations of module were considered outliers.\
                 \n\n{} is the most suspicious day.\
                 \n{:.2f}% of observations of that day were considered outliers.\
                 \n\nCheck out the returned dataframe with anomalous observations.\n--------------------\
                 """.format(report_obj.bad_module, report_obj.bad_module_rate,
                            report_obj.bad_day.strftime('%Y-%m-%d'),
                            report_obj.bad_day_rate))
    else:
        report_obj = Report()
        print("""\nREPORT\
                    \n--------------------\
                    \nCongrats, your modules work just fine. :)""")

    return report_obj


def plot_anomaly(df_long):
    df_long_anomaly = df_long[df_long['outlier'] == 1]
    df_long['SMA_scores'] = np.nan
    for i in df_long_anomaly['variable'].unique():
        if df_long_anomaly[df_long_anomaly['variable'] == i]['variable'].count() > 1:
            df_long['SMA_scores'] = df_long['SMA_scores'].fillna(df_long[df_long['variable'] == i]
                                                                 ['outlier_score'].rolling(window=7).mean())
    fig = px.line(df_long.dropna(), x='created_on', y='SMA_scores', hover_data=df_long.columns, color='variable')
    plotly.offline.plot(fig)
    return


def tidy_anomaly(data: Union[DataFrame, str, None] = None, train_size: float = 0.8, outliers_rate: float = 0.003,
                 classifier: str = 'K Nearest Neighbors', plot_a: bool = False):
    """ Return df containing outlier scores, outlier tags and report.

        Performs preprocessing, feature engineering on dataset and trains anomaly detection model on
        randomly selected subset of dataset. Returns subset of df that was considered anomalous with report.

        Parameters
            ----------
            data: pd.DataFrame.

            train_size: proportion of dataset to be used for training anomaly detection model.

            outliers_rate: proportion of training set to be considered outlier.

            classifier: string representing name of anomaly detection algorithm.

            plot_a: plots 2d contourf of anomaly detection scores.

        Returns
            -------
            df_modules_long_anomaly: pd.DataFrame with subset of dataset with measurements considered anomalous.

            report_obj: object containing elements of anomaly evaluation.

        """
    if isinstance(data, DataFrame):
        df = data
    elif isinstance(data, str):
        df = get_se_as_df(data)
    elif data is None:
        get_se_daily()
        df = get_se_as_df('se_daily.json')
    else:
        raise TypeError('Wrong data type.')

    df = drop_missing(df)

    yield_modules = get_yield(df, 'Module', 'String')

    df_modules = df[[i for i in df if 'Module' in i]]
    df_modules_long = pd.melt(df_modules.reset_index(), id_vars='created_on')
    yield_modules_long = pd.melt(yield_modules.reset_index(), id_vars='created_on')
    df_modules_long = df_modules_long.assign(yields=yield_modules_long['value'])

    y_pred, y_scores = detect_anomaly(df_modules_long[['value', 'yields']],
                                      train_size, outliers_rate, classifier, plot_a)

    df_modules_long = df_modules_long.assign(outlier=y_pred)
    df_modules_long = df_modules_long.assign(outlier_score=y_scores)

    anomaly_rate = round(df_modules_long[df_modules_long['outlier'] == 1].shape[0] / df_modules_long.shape[0] * 100, 2)

    print("""\nClassifier assumed {:.2f}% of outliers in randomly selected {:.0f}% of data from dataset.\
             \nAnomaly rate of all modules in dataset is {:.2f}%.
             """.format(outliers_rate * 100, train_size * 100, anomaly_rate, ))

    reporting = report_anomaly(df_modules_long)
    plot_anomaly(df_modules_long)

    return df_modules_long.sort_values('outlier_score', ascending=False), reporting


def tidy_individual_anomaly(data: Union[DataFrame, str, None] = None, train_size: float = 0.8,
                            outliers_rate: float = 0.0046, classifier: str = 'K Nearest Neighbors',
                            plot_a: bool = False):
    """ Return df containing outlier scores, outlier tags and report.

        Performs preprocessing, feature engineering on dataset and trains anomaly detection model on
        randomly selected subset of dataset. Returns subset of df that was considered anomalous with report.

        Parameters
            ----------
            data: pd.DataFrame.

            train_size: proportion of dataset to be used for training anomaly detection model.

            outliers_rate: proportion of training set to be considered outlier.

            classifier: string representing name of anomaly detection algorithm.

            plot_a: plots 2d contourf of anomaly detection scores.

        Returns
            -------
            df_modules_long_anomaly: pd.DataFrame with subset of dataset with measurements considered anomalous.

            report_obj: object containing elements of anomaly evaluation.

        """
    if isinstance(data, DataFrame):
        df = data
    elif isinstance(data, str):
        df = get_se_as_df(data)
    elif data is None:
        get_se_daily()
        df = get_se_as_df('se_daily.json')
    else:
        raise TypeError('Wrong data type.')

    df = drop_missing(df)

    df_modules = df[[i for i in df if 'Module' in i]]
    df_strings = df[[i for i in df if 'String' in i]]
    # df_inverters = df[[i for i in df if 'Inverter' in i]]

    yield_modules = get_yield(df_modules, df_strings)

    df_modules_long = pd.melt(df_modules.reset_index(), id_vars='created_on')
    yield_modules_long = pd.melt(yield_modules.reset_index(), id_vars='created_on')
    df_modules_long = df_modules_long.assign(yields=yield_modules_long['value'])

    df_modules_long['outlier_score'] = np.nan
    df_modules_long['outlier'] = np.nan

    for i in df_modules_long['variable'].unique():
        print(i)
        df_new = df_modules_long[df_modules_long['variable'] == i][['value', 'yields']]
        y_pred, y_scores = detect_anomaly(df_new, train_size=0.8, outliers_rate=0.0046,
                                                    classifier="K Nearest", plot=False)
        df_modules_long['outlier_score'] = df_modules_long['outlier_score'].fillna(pd.Series(y_scores))
        df_modules_long['outlier'] = df_modules_long['outlier'].fillna(pd.Series(y_pred))

    anomaly_rate = round(df_modules_long[df_modules_long['outlier'] == 1].shape[0] / df_modules_long.shape[0] * 100, 2)

    print("""\nClassifier assumed {:.2f}% of outliers in randomly selected {:.0f}% of data from dataset.\
             \nAnomaly rate of all modules in dataset is {:.2f}%.
             """.format(outliers_rate * 100, train_size * 100, anomaly_rate, ))

    reporting = report_anomaly(df_modules_long)
    #plot_anomaly(df_modules_long)

    return df_modules_long.sort_values('outlier_score', ascending=False), reporting


if __name__ == '__main__':
    filename = '../se_daily.json'
    df = get_se_as_df(filename)
    anomaly_df, report = tidy_anomaly(df)
    print(anomaly_df.head())
