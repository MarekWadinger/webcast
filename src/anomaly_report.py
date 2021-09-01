import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from scipy.stats import stats
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
from pyod.models.copod import COPOD
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM

import json
# from src.se_api_process import get_se_as_df


class Report:
    pass


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


def drop_missing(df: pd.DataFrame, valid_col_rate: float = 0.1) -> pd.DataFrame:
    """ Drop invalid columns and rows with missing values.

    Parameters
        ----------
        df: pd.DataFrame

        valid_col_rate: float representing the maximum % proportion of NAs in column
                        to be still considered valid.

    Returns
        -------
        pd.DataFrame with invalid columns and NA entries dropped from it.

    """
    if valid_col_rate >= 1:
        valid_col_rate = valid_col_rate / 100

    sum_na = df.isna().sum()
    cols_drop = sum_na[sum_na / df.shape[1] > valid_col_rate].index.values
    df = df.drop(columns=cols_drop)
    df = df.dropna()

    return df


def get_yield(df_subset: pd.DataFrame, df_set: pd.DataFrame) -> pd.DataFrame:
    """ Return product of row-wise division.

    Perform row-wise division between transposed subset and set to which it belongs.

    Parameters
        ----------
        df_subset: pd.DataFrame with column names containing: ".*[set_specific_string]\..*".

        df_set: pd.DataFrame with column names in format: ".*\s[set_specific_string]".

    Returns
        -------
        pd.DataFrame with product of row-wise division with the same shape as df_subset.

    """
    yield_modules = []
    # Works with arrays
    for string_id in df_set:
        string_num = string_id.split(' ')[1] + '.'
        # Returns array of modules belonging to given string
        modules = df_subset[[module for module in df_subset if string_num in module]].T.values
        temp_yield = np.divide(modules, df_set[string_id].values)
        yield_modules = [*yield_modules, *temp_yield]
    # Transforms array to DataFrame
    yield_modules = pd.DataFrame(np.transpose(yield_modules), index=df_subset.index, columns=df_subset.columns)

    return yield_modules


def plot_outlier_detection(df_floats: pd.DataFrame, y_pred: np.ndarray, clf, clf_name: str = None):
    """ Return contourf plot of the anomaly detection model.

        Plots contourf plot of decision scores marking area where datapoints would be considered inliers.

        Parameters
            ----------
            df_floats: pd.DataFrame with elements as floats.

            y_pred: numpy array of the same length as df_floats that assigns 0/1 (inlier/outlier) to each observation
                    according to fitted model.

            clf: fitted model.

            clf_name: name of fitted model.

        Returns
            -------
            Contourf plot

        """
    if df_floats.shape[1] > 2:
        print('Plotting first two variables')
    elif df_floats.shape[1] < 2:
        print('Sorry can not plot less than one variable')

    x_lim_min = df_floats.iloc[:, 0].min()
    x_lim_max = df_floats.iloc[:, 0].max()
    y_lim_min = df_floats.iloc[:, 1].min()
    y_lim_max = df_floats.iloc[:, 1].max()
    x_delta = 0.05 * (x_lim_max - x_lim_min)
    y_delta = 0.05 * (y_lim_max - y_lim_min)

    # predict raw anomaly score
    scores_pred = clf.decision_function(df_floats.iloc[:, [0, 1]]) * -1
    # threshold value to consider a datapoint inlier or outlier
    threshold = stats.scoreatpercentile(scores_pred, 100 * len(y_pred[y_pred == 1]) / len(y_pred))
    # coordinate array for vectorized evaluation of raw anomaly scores
    # TODO: Coarser grid sometimes returns error when plotting (threshold out of [zz.min();zz.max()]).
    xx, yy = np.meshgrid(np.linspace(x_lim_min - x_delta, x_lim_max + x_delta, 1000),
                         np.linspace(y_lim_min - y_delta, y_lim_max + y_delta, 1000))
    # decision function calculates the raw anomaly score for every point
    zz = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
    zz = zz.reshape(xx.shape)
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

    #
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


def find_anomaly(df_floats: pd.DataFrame, train_size: float, outliers_fraction: float, classifier: str = 'IForest',
                 plot_a: bool = True):
    """ Return binary classified outlier and raw outlier score.

    Performs training of anomaly detection model on subset of dataset and returns
    binary label and decision score for whole dataset.

    Parameters
        ----------
        df_floats: pd.DataFrame with elements as floats.

        train_size: proportion of dataset to be used for training anomaly detection model.

        outliers_fraction: proportion of training set to be considered outlier.

        classifier: string representing name of anomaly detection algorithm.

        plot_a: plots 2d contourf of anomaly detection scores.

    Returns
        -------
        y_pred: numpy array of the same length as df_floats that assigns 0/1 (inlier/outlier) to each observation
                    according to fitted model.
        y_scores: numpy array of the same length as df_floats that assigns outlier scores to each observation
                    according to fitted model.

        clf_name: name of fitted model

    """
    if train_size > 1:
        train_size = train_size / 100
    # TODO: Find out empirical way to set contamination level
    if outliers_fraction >= 1:
        outliers_fraction = outliers_fraction / 100

    random_state = np.random.RandomState(42)

    # TODO: Perform scaling of data for AKNN, CBLOF, HBOS, KNN, OCSVM
    classifiers = {
        'Average KNN (AKNN)': KNN(method='mean', contamination=outliers_fraction),
        'Cluster-based Local Outlier Factor (CBLOF)': CBLOF(contamination=outliers_fraction, check_estimator=False,
                                                            random_state=random_state),
        'Copula based Outlier Detection (COPOD)': COPOD(contamination=outliers_fraction),
        'Histogram-base Outlier Detection (HBOS)': HBOS(contamination=outliers_fraction),
        'Isolation Forest (IForest)': IForest(contamination=outliers_fraction, random_state=random_state),
        'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
        'One-Class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
        'Principal component analysis (PCA)': PCA(contamination=outliers_fraction)
    }

    x_train, x_test = train_test_split(df_floats, train_size=train_size)

    clf_name = ''
    for name in classifiers.keys():
        if classifier in name:
            clf_name = name
            break

    if clf_name:
        clf = classifiers.get(clf_name)
        clf.fit(x_train)
        y_pred = clf.predict(df_floats)  # binary labels (0: inliers, 1: outliers)
        y_scores = clf.decision_function(df_floats)  # raw outlier scores
    else:
        raise NameError("Unknown classifier. "
                        "Please use one of those: {}.".format(list(classifiers.keys())))

    # for i, (clf_name, clf) in enumerate(classifiers.items()):
    #    scaler = MinMaxScaler(feature_range=(0, 1))
    #    df_floats.iloc[:, [0, 1]] = scaler.fit_transform(df_floats.iloc[:, [0, 1]])
    #    # fit model
    #    clf.fit(df_floats)
    #    # prediction of a datapoint category outlier or inlier
    #    y_pred = clf.predict(df_floats)

    if plot_a:
        plot_outlier_detection(df_floats, y_pred, clf, clf_name)

    return y_pred, y_scores, clf_name


def anomaly_report(df: pd.DataFrame, train_size: float = 0.8, outliers_rate: float = 0.05,
                   classifier: str = 'IForest', plot_a: bool = True):
    """ Return subset of df containing outliers and report.

        Performs preprocessing, feature engineering on dataset and trains anomaly detection model on
        randomly selected subset of dataset. Returns subset of df that was considered anomalous with report.

        Parameters
            ----------
            df: pd.DataFrame.

            train_size: proportion of dataset to be used for training anomaly detection model.

            outliers_rate: proportion of training set to be considered outlier.

            classifier: string representing name of anomaly detection algorithm.

            plot_a: plots 2d contourf of anomaly detection scores.

        Returns
            -------
            df_modules_long_anomaly: pd.DataFrame with subset of dataset with measurements considered anomalous.

            report_obj: object containing elements of anomaly evaluation.

        """
    df = drop_missing(df)

    df_strings = df[[i for i in df if 'String' in i]]
    df_modules = df[[i for i in df if 'Module' in i]]
    # df_inverter = df[['Inverter 1']]

    yield_modules = get_yield(df_modules, df_strings)

    df_modules_long = pd.melt(df_modules.reset_index(), id_vars='created_on')
    yield_modules_long = pd.melt(yield_modules.reset_index(), id_vars='created_on')
    df_modules_long = df_modules_long.assign(yields=yield_modules_long['value'])

    y_pred, y_scores, clf_name = find_anomaly(df_modules_long[['value', 'yields']],
                                              train_size, outliers_rate, classifier, plot_a)

    df_modules_long = df_modules_long.assign(outlier=y_pred)
    df_modules_long = df_modules_long.assign(outlier_score=y_scores)

    df_modules_long_anomaly = df_modules_long[df_modules_long['outlier'] == 1]

    report_obj = Report()
    report_obj.anomaly_rate = round(df_modules_long_anomaly.shape[0] / df_modules_long.shape[0] * 100, 2)

    report_obj.bad_module = df_modules_long_anomaly.groupby(['variable']).count()['outlier'].idxmax()
    sum_bad_module_anomaly = df_modules_long_anomaly[df_modules_long_anomaly['variable'] ==
                                                     report_obj.bad_module].shape[0]
    sum_bad_module = df_modules_long[df_modules_long['variable'] == report_obj.bad_module].shape[0]
    report_obj.bad_module_rate = round(sum_bad_module_anomaly / sum_bad_module * 100, 2)

    report_obj.bad_day = df_modules_long_anomaly.groupby(['created_on']).count()['outlier'].idxmax()
    sum_bad_day_anomaly = df_modules_long_anomaly[df_modules_long_anomaly['created_on'] ==
                                                  report_obj.bad_day].shape[0]
    sum_bad_day = df_modules_long[df_modules_long['created_on'] == report_obj.bad_day].shape[0]
    report_obj.bad_day_rate = round(sum_bad_day_anomaly / sum_bad_day * 100, 2)

    print("""\nREPORT\
            \n--------------------\
            \nUsed classifier: {}\
            \nModel assumed {:.2f}% of outliers in randomly selected {:.0f}% of data from dataset.\
            \nAnomaly rate of all modules in dataset is {:.2f}%. \nMost critical module is the {}.\
            \nwith anomaly rate of {:.2f}%.\
            \nDay that should be examined is {}.\
            \nwith anomaly rate of {:.2f}%.\n--------------------\
            """.format(clf_name, outliers_rate * 100, train_size * 100, report_obj.anomaly_rate, report_obj.bad_module,
                       report_obj.bad_module_rate, report_obj.bad_day.strftime("%Y-%m-%d"),
                       report_obj.bad_day_rate))

    return df_modules_long_anomaly, report_obj


if __name__ == '__main__':
    filename = '../se_daily.json'
    df = get_se_as_df(filename)
    anomaly_df, report = anomaly_report(df)
    print(anomaly_df.head())