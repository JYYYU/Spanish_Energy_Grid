import pandas as pd
import numpy as np
import holidays
import itertools
import random
import time

from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss, ccf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels import robust
from scipy.stats import skew, kurtosis
from scipy.interpolate import interp1d
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from keras.regularizers import l2

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

def time_spacing(df):

    try:
        print(f"This dataframe has {len(df)} rows.")
        return df.index.to_series().diff().value_counts()
    except:
        raise ValueError("df must be a pandas DataFrame")

def missing_features(df):

    try:
        total = df.isna().sum()
        missing = total[total > 0]
        percentage = (missing/len(df)) * 100
        return pd.merge(percentage.to_frame(name="Missing (%)"), missing.to_frame(name="Missing (Count)"), left_index=True, right_index=True, how="outer")
    except:
        raise ValueError("df must be a pandas DataFrame")

def holiday_dates(years, code="ES"):

    if all(isinstance(year, int) for year in years) and isinstance(code, str):
        try:
            holidays_dict = holidays.country_holidays(code, years=years)
            return list(holidays_dict.keys())
        except:
            raise ValueError("Enter proper country code; Refer to: https://pypi.org/project/holidays/")
    else:
        raise ValueError("Enter proper list of years or enter proper country code; Refer to: https://pypi.org/project/holidays/")
    
def find_outliers(series):
        
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
        
    return series[(series < lower_bound) | (series > upper_bound)]

def upperbound_outliers(series):

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
        
    return series[(series > upper_bound)]

def outlier_count(series, func=find_outliers):

    results = func(series)
    
    return len(results)

def robust_zscore(series):

    median = np.median(series)
    mad = robust.mad(series)
    mod_zscores = 0.6745 * (series - median) / mad
    return mod_zscores

def detect_anomalies_zscore(series, threshold=2):
    mean = series.mean()
    std_dev = series.std()
    z_scores = (series - mean) / std_dev
    return (z_scores.abs() > threshold).astype(int)
    
def detect_anomalies_percentile(series, lower_percentile=0.05, upper_percentile=0.95):
    lower_bound = series.quantile(lower_percentile)
    upper_bound = series.quantile(upper_percentile)
    return ((series < lower_bound) | (series > upper_bound)).astype(int)

def custom_stats(grouped_df):

    aggregated_data = grouped_df.agg(count='count', mean='mean', median='median',
                                      std='std', min='min', max='max',
                                      skewness=skew, 
                                      kurtosis=kurtosis, 
                                      outliers_count= outlier_count).reset_index() 
    
    return aggregated_data

def decompose_ts(df, model, period=24):
    
    temp = df.copy()
    decomp = seasonal_decompose(temp, model, period=period)
    
    return decomp

def check_duplicates_ts(df):

    def check_dups_in_group(group):
            return not group.eq(group.iloc[0]).all(axis=1).all()
    
    duplicates_with_same_datetime = df.groupby(df.index).apply(check_dups_in_group)
    datetime_indexes_with_differing_rows = duplicates_with_same_datetime[duplicates_with_same_datetime].index
    return datetime_indexes_with_differing_rows.to_list()

def maintain_feature_order(reference_columns, find_columns):

    sorted_columns = [col for col in reference_columns if col in find_columns]
    
    return sorted_columns

# Ensures that the index is sorted 
def save_file(df, path, index=True):

    if not df.index.is_monotonic_increasing:
        df.sort_index(inplace=True)
    df.to_csv(path, index=index)
    print(f"File is saved.")

# df1 and df2 are respectively the Energy and Weather Features datasets
def merge_energy_weather(df1, df2):

    final_df = df1.copy()

    if "city_name" not in df2.columns:
        raise ValueError(f"The weather feature data set does not contain the column 'city_name'. Please provide a dataset with that column.")
    else:
        for df in [final_df, df2]:
            if df.index.name != "time":
                df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("CET")
                df.set_index("time", inplace=True)
        
        for city, city_df in df2.groupby("city_name"):
            city_df.drop(["city_name"], axis=1, inplace=True)
            city_df.columns = ["".join([col.replace(" ", "_"), "_", city]) for col in city_df.columns]
            final_df = pd.merge(final_df, city_df, left_index=True, right_index=True)
        
        return final_df

# FOR WEATHER FEATURES
def wf_groupby_stats(df, target, groupby_list):

    temp = is_weekday(df.copy())
    temp = is_holidays(temp.copy(), temp.index.year.unique())
    temp = add_seasons(temp.copy())
    temp = add_time_features(temp.copy())
    grouped_data = temp.groupby(groupby_list)[target]

    return custom_stats(grouped_data)

# FOR WEATHER FEATURES
def avg_wf_by_time(df, target, time_groupby_list):

    temp = add_seasons(df.copy())
    temp = add_time_features(temp.copy())
    grouped_data = pd.DataFrame(temp.groupby(time_groupby_list)[target].mean())

    return grouped_data

# FOR WEATHER FEATURES
def wf_count_outliers_by_cities(df, cities, target, groupby_feature):

    for city in cities:
        print(f"For {city}, the number of large outliers per {groupby_feature}:")
        city_df = df[(df["city_name"]==city)].copy()
        city_count = []
        for feature in df[groupby_feature].unique():
            feature_total = len(upperbound_outliers(city_df[city_df[groupby_feature]==feature][target]))
            city_count.append(feature_total)
            print(f"{feature}:{feature_total}")
        print(f"*{city} has a total of {sum(city_count)} large outliers.*")
        print("__________________________________________________________")

def function_timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  
        result = func(*args, **kwargs)  
        end_time = time.time()  
        print(f"Function '{func.__name__}' took {end_time - start_time:.4f} seconds to execute.")
        return result
    return wrapper

#####################################################################################
# Stationary Test #
#####################################################################################

def is_stationary_adf(df, regression='c', alpha=0.05):
    result = adfuller(df.copy(), regression=regression)
    p_val  = result[1]
    if p_val <= alpha:
        print("Time series is stationary (ADF Test).")
        return f"The p-value is {p_val}."
    else:
        print("Time series is not stationary (ADF Test).")
        return f"The p-value is {p_val}."

def is_stationary_kpss(df, regression='c', alpha=0.05):
    result = kpss(df.copy(), regression=regression)
    p_val = result[1]
    if p_val <= alpha:
        print("Time series is not stationary (KPSS Test).")
        return f"The p-value is {p_val}."
    else:
        print("Time series is stationary (KPSS Test).")
        return f"The p-value is {p_val}."  
    
#####################################################################################
# Baseline Model #
#####################################################################################

def baseline_model(train_data, test_data, order, seasonal_order, original_series):

    predictions = sarima_walk_forward_validation(train_data, test_data, order, seasonal_order)

    # Transform Data Back
    pred_series = predictions + np.log(1 + original_series.shift(1))
    pred_series = np.exp(pred_series.loc[~pred_series.isna()]) -1

    index_date = pred_series.index[0]
    index_train = index_date - pd.Timedelta(days=100)
    train_data = original_series[(original_series.index >= index_train) & (original_series.index < index_date)]
    test_data = original_series.loc[index_date:]
    
    plt.figure(figsize=(15, 6))
    plt.plot(train_data.index, train_data.values, label="Train", lw=0.5, alpha=0.5, color="green")
    plt.plot(test_data.index, test_data.values, label="Test", lw=1.5, alpha=0.7, color="yellow")
    plt.plot(pred_series.index, pred_series.values, label="Forecast", lw=0.3, alpha=0.9, color="purple")

    plt.title("Baseline Model")
    plt.xlabel("Time")
    plt.ylabel("Total Load Actual")
    plt.legend()
    plt.tight_layout()
    plt.show()

    pred_df = pred_series.to_frame(name="Results")

    return evaluate_model(pred_series, test_data, False), pred_df

def sarima_walk_forward_validation(train_data, test_data, order, seasonal_order):

    history = list(train_data.copy())
    predictions = []

    model = ARIMA(history, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()

    for i in range(len(test_data)):
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test_data[i])
        model_fit = model_fit.append([test_data[i]], refit=False)

    return pd.Series(predictions, index=test_data.index)

def baseline_model_v1(target_data, params, test_size, original_series):
    
    predictions = arima_walk_forward_validation(target_data, params, test_size)

    # Transform Data Back
    pred_series = predictions + np.log(1 + original_series.shift(1))
    pred_series = np.exp(pred_series.loc[~pred_series.isna()]) -1

    # Plot Baseline Model
    # Note: Will plot 100 days of the training set
    index_date = pred_series.index[0]
    index_train = index_date - pd.Timedelta(days=100)
    train_data = original_series[(original_series.index >= index_train) & (original_series.index < index_date)]
    test_data = original_series.loc[index_date:]
    
    plt.figure(figsize=(15, 6))

    plt.plot(train_data.index, train_data.values, label="Train", lw=0.5, alpha=0.5, color="green")
    plt.plot(test_data.index, test_data.values, label="Test", lw=1.5, alpha=0.7, color="yellow")
    plt.plot(pred_series.index, pred_series.values, label="Forecast", lw=0.3, alpha=0.9, color="purple")

    plt.title("Baseline Model")
    plt.xlabel("Time")
    plt.ylabel("Total Load Actual")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return evaluate_model(pred_series, test_data, False)

def arima_walk_forward_validation(target_data, params, test_size):

    train_index = int(len(target_data) * (1 - test_size))
    train, test = target_data[:train_index], target_data[train_index:]
    history = list(train.copy())
    predictions = []

    model = ARIMA(history, order=params)
    model_fit = model.fit()

    for i in range(len(test)):
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[i])
        model_fit = model_fit.append([test[i]], refit=False)

    return pd.Series(predictions, index=test.index)
    
#####################################################################################
# Plotting Helpers #
#####################################################################################

def get_n_colours(n, remove_colors=[]):
    count, colour_list = 0, []
    remove_colors = ['antiquewhite','floralwhite','ghostwhite','navajowhite',
                     'white','whitesmoke', "snow","seashell","ivory","beige",
                     "honeydew", "azure", "aliceblue"] + remove_colors
    colour_names = list(mcolors.CSS4_COLORS.keys())
    colour_names = list(filter(lambda x: x not in remove_colors, colour_names))

    return random.sample(colour_names, n)

# USED FOR TOTAL LOAD ACTUAL
def plot_tv_four_weeks(df, target):

    temp = df.copy()
    inital_time = temp.index.min()
    four_weeks = temp.loc[inital_time:inital_time+pd.Timedelta(weeks=4)].copy()

    plt.figure(figsize=(12, 6))
    plt.plot(four_weeks.index, four_weeks[target], label=target, lw=0.75)

    missing_index = four_weeks[four_weeks[target].isna()].index
    plt.scatter(missing_index, [np.min(four_weeks[target])]*len(missing_index), color='red', label='Hours with Missing Data', marker='x', s=25, lw=0.75)

    plt.xlabel('Time')
    plt.ylabel('Total Load Actual (MW)')
    plt.title('Four Weeks of Total Load Actual')
    plt.legend()
    plt.grid(True)
    plt.show()

# USED FOR TOTAL LOAD ACTUAL
def plot_monthly_avg_load_per_year(df, target):

    temp = df.copy()
    monthly = temp[[target]].copy()
    monthly["year"] = monthly.index.year
    monthly["month"] = monthly.index.month   
    monthly_avg = monthly.groupby(["year", "month"]).mean().reset_index()
    pivot_table = monthly_avg.pivot(index='month', columns='year', values=target)

    plt.figure(figsize=(10, 6))
    for col in pivot_table:
        plt.plot(pivot_table.index, pivot_table[col], label=col, lw=0.5, alpha=0.5)
    plt.suptitle("Monthly Average Total Load Actual Per Year")
    plt.xlabel("Months")
    plt.ylabel("Total Load Actual (MW)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# USED FOR TOTAL LOAD ACTUAL
def plot_monthly_avg_load(df, target):

    temp = df.copy()
    monthly = temp[[target]].copy()
    monthly["year"] = monthly.index.year
    monthly["month"] = monthly.index.month   
    monthly_avg = monthly.groupby(["year","month"]).mean().reset_index()
    pivot_table = monthly_avg.pivot(index='month', columns='year', values=target)

    stacked = pivot_table.reset_index().melt(id_vars=["month"], var_name="year", value_name=target)
    stacked["year_month"] = stacked["year"].astype(str) + '-' + stacked["month"].astype(str).str.zfill(2)
    x_labels = stacked['year_month']
    x_tick_positions = range(0, len(x_labels), len(x_labels) // 13)  
    x_tick_labels = [x_labels[i] for i in x_tick_positions] 

    plt.figure(figsize=(10, 6))
    
    plt.plot(stacked["year_month"], stacked[target], label=target, lw=0.5, alpha=0.5)
    plt.xticks(ticks=x_tick_positions, labels=x_tick_labels, rotation=45)
    plt.suptitle("Monthly Average Total Load Actual")
    plt.xlabel("Months")
    plt.ylabel("Total Load Actual (MW)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# USED FOR TOTAL LOAD ACTUAL
def plot_energy_comp_target(df, target):

    temp = is_holidays(df.copy(), df.index.year.unique())
    temp = is_weekday(temp)

    fig, axes = plt.subplots(1, 4, figsize=(10, 6), sharey=True)

    fig.suptitle("Total Load Actual (MW) by Type of Days")

    # Target on Holidays
    sns.boxplot(temp[temp["isHoliday"] == 0], y=target, ax=axes[0], color="purple") 
    axes[0].set_xlabel('Non-Holidays')
    axes[0].set_ylabel('Total Load Actual (MW)') 

    # Target on Holidays
    sns.boxplot(temp[temp["isHoliday"] == 1], y=target, ax=axes[1], color="red") 
    axes[1].set_xlabel('Holidays')
    axes[1].set_ylabel('') 

    # Target on Weekends
    sns.boxplot(temp[temp["isWeekday"] == 0], y=target, ax=axes[2], color="blue") 
    axes[2].set_xlabel('Weekends')
    axes[2].set_ylabel('') 

    # Target on Holidays
    sns.boxplot(temp[temp["isWeekday"] == 1], y=target, ax=axes[3], color="green") 
    axes[3].set_xlabel('Weekdays')
    axes[3].set_ylabel('') 

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, left=0.12)
    plt.show()

# USED FOR TOTAL LOAD ACTUAL
def plot_energy_dist_target(df, target):

    temp = is_holidays(df.copy(), df.index.year.unique())
    temp = is_weekday(temp)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].hist(temp[temp["isHoliday"] == 0][target], bins=30, color='blue', edgecolor='black')
    axes[0, 0].set_title('Non-Holidays')
    axes[0, 0].set_xlabel('Total Actual Load (MW)')
    axes[0, 0].set_ylabel('Frequency')

    axes[0, 1].hist(temp[temp["isHoliday"] == 1][target], bins=30, color='green', edgecolor='black')
    axes[0, 1].set_title('Holidays')
    axes[0, 1].set_xlabel('Total Actual Load (MW)')
    axes[0, 1].set_ylabel('Frequency')

    axes[1, 0].hist(temp[temp["isWeekday"] == 0][target], bins=30, color='red', edgecolor='black')
    axes[1, 0].set_title('Weekends')
    axes[1, 0].set_xlabel('Total Actual Load (MW)')
    axes[1, 0].set_ylabel('Frequency')

    axes[1, 1].hist(temp[temp["isWeekday"] == 1][target], bins=30, color='purple', edgecolor='black')
    axes[1, 1].set_title('Weekdays')
    axes[1, 1].set_xlabel('Total Actual Load (MW)')
    axes[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# USED FOR TOTAL LOAD ACTUAL
def plot_missing_by_hour(df, target):

    temp = is_holidays(df.copy(), df.index.year.unique())
    temp['hour'] = temp.index.hour

    cond1 = temp[temp[target].isna()]
    cond2 = temp[(temp[target].isna()) & (temp["isHoliday"] == 1)] 

    print(f"The number of rows in the target variable where there is missing data is {len(cond1)}")
    print(f"The number of rows in the target variable where there is missing data on a holiday is {len(cond2)}")
    missing_by_hour = temp.groupby("hour")[[target]].apply(lambda x: x.isna().sum())

    plt.figure(figsize=(10, 6))
    plt.hist(missing_by_hour.index, bins=np.arange(0,25)-0.5, weights=missing_by_hour, edgecolor='k', alpha=0.7)
    plt.title("Distribution of missing 'Total Load Actual' by Hours")
    plt.xlabel('Hours')
    plt.ylabel('Frequency')
    plt.xticks(ticks=missing_by_hour.index, labels=missing_by_hour.index)
    plt.show()

# USED FOR TOTAL LOAD ACTUAL
def plot_tv_by_the_hour(df, target, i_confint=False):

    temp = is_holidays(df.copy(), df.index.year.unique())
    temp['hour'] = temp.index.hour

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    sns.boxplot(data=temp[temp["isHoliday"] == 0], x="hour", y=target, ax=axes[0])
    axes[0].set_title("Total Load Actual by hour (No Holidays)")

    sns.boxplot(data=temp, x="hour", y=target, ax=axes[1])
    axes[1].set_title("Total Load Actual by hour")

    acf_data = temp[target].dropna()
    acf_values, confint = acf(acf_data, nlags=170, alpha=0.05)
    axes[2].plot(range(len(acf_values)), acf_values, marker=".", linestyle="-", color="red")
    if i_confint:
        confint_lower = confint[:, 0] - acf_values
        confint_upper = confint[:, 1] - acf_values
        axes[2].fill_between(range(len(acf_values)), confint_lower, confint_upper, color='lightgrey', alpha=0.7)
    axes[2].vlines(x=[24*i for i in range(1,8)], ymin=min(acf_values), ymax=max(acf_values), lw=0.5, ls= "--", color="blue", alpha=0.6)
    axes[2].set_title("Autocorrelation")
    axes[2].set_xlabel('Lags')
    axes[2].set_ylabel('ACF')

    plt.tight_layout()
    plt.show()

# USED FOR TOTAL LOAD ACTUAL
def plot_lags_n_days(df, target, lags, days=2):

    temp = df.copy()
    inital_time = temp.index.min()
    trimmed_df = temp.loc[inital_time:inital_time+pd.Timedelta(days=days)].copy()

    series = trimmed_df[target]

    plt.figure(figsize=(14, 8))
    for lag in lags:
        if lag == 0:
            plt.plot(series.shift(lag), label=target, lw=0.75)
        else:
            plt.plot(series.shift(lag), label=f'Lag {lag}', lw=0.75)
    
    plt.xlabel('Time')
    plt.ylabel('Total Load ACtual (MW)')
    plt.title('Total Load Actual with Lags')
    plt.legend()
    plt.grid(True)
    plt.show()

# USED FOR TOTAL LOAD ACTUAL
def plot_first_n_days(df, target, days=2):

    temp = df.copy()
    inital_time = temp.index.min()
    trimmed_df = temp.loc[inital_time:inital_time+pd.Timedelta(days=days)].copy()

    series = trimmed_df[target]
    rolling_avg_24hr = series.rolling(window=24).mean()
    rolling_avg_7_days = series.rolling(window=24*7).mean()

    plt.figure(figsize=(20, 8))
    plt.plot(series, label=target, lw=0.75, alpha=0.7)
    plt.plot(rolling_avg_24hr, label="Rolling 24 Hour Average", color="red")
    plt.plot(rolling_avg_7_days, label="Rolling Weekly Average", color="cyan")
    plt.xlabel('Time')
    plt.ylabel('Total Load Actual (MW)')
    plt.title(f'Total Load Actual for first {days} days')
    plt.legend()
    plt.grid(True)
    plt.show()
    
# USED FOR GENERATION FEATURES
def plot_energy_histograms(df, columns, bins=100):

    num_cols = len(columns)
    plots_per_row = 2
    num_rows = (num_cols + plots_per_row - 1) // plots_per_row

    fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(10, 5 * num_rows))
    axes = axes.flatten() 

    for i, column in enumerate(columns):
        df[column].plot(kind='hist', bins=bins, ax=axes[i], title=f"Distribution of \n {column}")
        axes[i].set_xlabel(f"{column} (MW)")

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# USED FOR GENERATION FEATURES
def plot_gen_boxplots(df, columns):

    num_cols = len(columns)
    plots_per_row = 2
    num_rows = (num_cols + plots_per_row - 1) // plots_per_row

    fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(10, 5 * num_rows))
    axes = axes.flatten() 

    for i, column in enumerate(columns):
        sns.boxplot(data=df[df["isHoliday"] == 0], ax=axes[i], x="hour", y=column)
        axes[i].set_title(column)
        axes[i].set_xlabel(f"Hour")
        axes[i].set_ylabel(f"Generation (MW)")
    
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# USED FOR GENERATION FEATURES
def plot_acf_pacf_group(df, columns, nlags=170, is_acf = True, i_confint=False):

    title_map = {True: "Autocorrelation", False: "Partial Autocorrelation"}
    func_map = {True: acf, False: pacf}
    super_title = title_map.get(is_acf)
    func = func_map.get(is_acf)
    
    num_cols = len(columns)
    plots_per_row = 2
    num_rows = (num_cols + plots_per_row - 1) // plots_per_row

    fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(10, 5 * num_rows))
    axes = axes.flatten() 

    for i, column in enumerate(columns):
        values, confint = func(df[column].copy().dropna(), nlags=170, alpha=0.05)
        axes[i].plot(range(len(values)), values, marker=".", linestyle="-", color="red")
        if i_confint:
            confint_lower = confint[:, 0] - values
            confint_upper = confint[:, 1] - values
            axes[i].fill_between(range(len(values)), confint_lower, confint_upper, color='lightgrey', alpha=0.7)
        bound = np.ceil(nlags/24).astype(int)
        if bound > 1:
            axes[i].vlines(x=[24*i for i in range(1,bound)], ymin=min(values), ymax=max(values), lw=0.5, ls= "--", color="blue", alpha=0.6)
        axes[i].set_title(column)
        axes[i].set_xlabel('Lags')
        axes[i].set_ylabel('ACF')
    
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(super_title)
    plt.tight_layout()
    plt.show()

# USED FOR WEATHER FEATURES
def plot_city_boxplots(dfs, column, titles, ylab):

    fig, axes = plt.subplots(1, len(dfs)//2 + 1, figsize=(10, 5))
    axes = axes.flatten()

    if len(dfs) == 1:
        sns.boxplot(data=dfs[0], ax=axes[0], y=column)
        plt.title(titles[0])
        plt.ylabel(ylab)

    else:
        for i, df in enumerate(dfs):
            sns.boxplot(data=dfs[i], ax=axes[i], y=column)
            axes[i].set_title(titles[i])
            axes[i].set_ylabel(ylab)

    plt.tight_layout()
    plt.show()

# USED FOR WEATHER FEATURES
def plot_find_city_trend(df, column, xlab, bins=100, i_confint=False):

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes = axes.flatten() 

    df[column].plot(kind='hist', bins=bins, ax=axes[0], title=f"Distribution of \n {column}")
    axes[0].set_xlabel(xlab)

    acf_values, confint = acf(df[column].copy().dropna(), nlags=170, alpha=0.05)
    axes[1].plot(range(len(acf_values)), acf_values, marker=".", linestyle="-", color="red")
    if i_confint:
        confint_lower = confint[:, 0] - acf_values
        confint_upper = confint[:, 1] - acf_values 
        axes[1].fill_between(range(len(acf_values)), confint_lower, confint_upper, color='lightgrey', alpha=0.7)
    axes[1].vlines(x=[24*i for i in range(1,8)], ymin=min(acf_values), ymax=max(acf_values), lw=0.5, ls= "--", color="blue", alpha=0.6)
    axes[1].set_title(f"Autocorrelation for {column}")
    axes[1].set_xlabel('Lags')
    axes[1].set_ylabel('ACF')

    plt.tight_layout()
    plt.show()

# USED FOR WEATHER FEATURES
def plot_no_errors(df, column, titles, xlabs, ylabs, bins=100):
    """
    Plots a histogram and boxplot
    """

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes = axes.flatten() 

    df[column].plot(kind='hist', bins=bins, ax=axes[0], title=titles[0])
    axes[0].set_xlabel(xlabs[0])

    sns.boxplot(data=df, ax=axes[1], y=column)
    axes[1].set_title(titles[1])
    axes[1].set_ylabel(ylabs[1])

    plt.tight_layout()
    plt.show()

# USED FOR WEATHER FEATURES
def plot_city_comp_bp(df, target, cities):

    fig, axes = plt.subplots(1, len(cities), figsize=(12, 7), sharey=True)
    axes = axes.flatten() 
    
    fig.suptitle(f"{target} by Cities")

    y_labs = [target] + list(np.repeat("", len(cities) - 1))

    for i, city in enumerate(cities):
        sns.boxplot(df[df["city_name"] == city], y=target, ax=axes[i])
        axes[i].set_xlabel(city)
        axes[i].set_ylabel(y_labs[i])  

    plt.tight_layout()
    plt.show()

# USED FOR WEATHER FEATURES    
def plot_wf_acf_pacf(df, target, nlags = 170, time_groupby_list=None, i_confint=False):
    
    if time_groupby_list != None:
        index, columns = time_groupby_list[0], time_groupby_list[1]
        temp = add_seasons(df.copy())
        temp = add_time_features(temp)
        grouped_data_by_time = avg_wf_by_time(temp.copy(), target, time_groupby_list).reset_index()
        grouped_data_by_time = grouped_data_by_time.pivot(index=index, columns=columns, values=target)
        plot_acf_pacf_group(grouped_data_by_time, grouped_data_by_time.columns, nlags=nlags, i_confint=i_confint)
    else:
        plot_acf(df, target, nlags=nlags, i_confint=i_confint)
        plot_pacf(df, target, nlags=nlags, i_confint=i_confint)
        
# USED TO UNDERSTAND INTERACTIONS BETWEEN WEEKDAYS, HOLIDAYS, AND BUSINESS HOURS 
# ON A FEATURE/VARIABLE
def plot_comp_target(df, target, ylab):
    
    temp = df.copy()

    if not all(col in temp.columns for col in ["is_business_hour", "isHoliday", "isWeekday"]):
        
        temp = add_business_hours(temp)
        temp = is_holidays(temp)
        temp = is_weekday(temp)

    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    color_dict = {0:"purple", 1:"red", 2:"green"}
    products = list(itertools.product([0,1],[1,2,0],[0,1]))

    for i in range(len(products)):
        
        weekday, business_hour, holiday = products[i]

        filtered_data = temp[
                        (temp["isWeekday"] == weekday) &
                        (temp["is_business_hour"] == business_hour) &
                        (temp["isHoliday"] == holiday)]
        
        sns.boxplot(filtered_data, y=target, ax=axes[i // 3, i % 3], color=color_dict[i % 3])

        axes[i // 3, i % 3].set_xlabel(
                f'{"Weekday" if weekday == 1 else "Weekend"} \n'
                f'During {"Business Hours" if business_hour == 1 else "Siesta Hours" if business_hour == 2 else "After Business Hours"} \n'
                f'and on a {"Holiday" if holiday == 1 else "Non-Holiday"}'
            )
        axes[i // 3, i % 3].set_ylabel(ylab)

    plt.tight_layout()
    plt.show()

# DICTIONARIES ARE OF LENGTH 1
def plot_target_v_weather(df, axes1_dict, axes2_dict, title="",  trendline_1=False, trendline_2=False, lw1=1.5, lw2=1.5):

    labels1, labels2 = list(axes1_dict.keys())[0], list(axes2_dict.keys())[0]
    colours = get_n_colours(len(labels1 + labels2))
    colour_count = 0
    
    fig, axes1 = plt.subplots(figsize=(20, 6))
    axes1_label = labels1

    for column in axes1_dict[axes1_label]:
        if trendline_1:
            resampled = df[column].resample("5D").first()
            resampled_index = (resampled.index.tz_localize(None) - pd.Timestamp("1970-01-01")) /  pd.Timedelta(seconds=1)
            linspace_x = np.linspace(resampled_index.min(), resampled_index.max(), len(resampled))
            energy_spline = interp1d(x=resampled_index, y = resampled.values)
            axes1.plot(resampled.index, energy_spline(linspace_x), label=f'{column} Linear Spline', lw=lw1, alpha=1, color='red')
        else:
            axes1.plot(df.index, df[column], label=column, lw=0.5, alpha=0.6, color=colours[colour_count])
            colour_count += 1
    axes1.set_xlabel('Time')
    axes1.set_ylabel(axes1_label)

    axes2 = axes1.twinx()
    axes2_label = labels2

    for column in axes2_dict[axes2_label]:
        if trendline_2:
            resampled = df[column].resample("5D").first()
            resampled_index = (resampled.index.tz_localize(None) - pd.Timestamp("1970-01-01")) /  pd.Timedelta(seconds=1)
            linspace_x = np.linspace(resampled_index.min(), resampled_index.max(), len(resampled))
            energy_spline = interp1d(x=resampled_index, y = resampled.values)
            axes2.plot(resampled.index, energy_spline(linspace_x), label=f'{column} Linear Spline', lw=lw2, alpha=1, color=colours[colour_count], ls="dotted")
            colour_count += 1
        else:
            axes2.plot(df.index, df[column], label=column, lw=0.5, alpha=0.5)
            colour_count += 1
    axes2.set_ylabel(axes2_label)

    fig.suptitle(title)
    lines, labels = axes1.get_legend_handles_labels()
    lines2, labels2 = axes2.get_legend_handles_labels()
    axes2.legend(lines + lines2, labels + labels2, loc='upper left')
    plt.tight_layout()
    plt.show()

# FOR PRESSURE EDA
def plot_target_v_weather_pressure(df, axes1_dict, axes2_dict, title="",  trendline_1=False, trendline_2=False, lw1=1.5, lw2=1.5):

    labels1, labels2 = list(axes1_dict.keys())[0], list(axes2_dict.keys())[0]
    colours = get_n_colours(len(labels1 + labels2))
    colour_count = 0
    
    fig, axes1 = plt.subplots(figsize=(20, 6))
    axes1_label = labels1

    for column in axes1_dict[axes1_label]:
        if trendline_1:
            resampled = df[column].resample("5D").first()
            resampled_index = (resampled.index.tz_localize(None) - pd.Timestamp("1970-01-01")) /  pd.Timedelta(seconds=1)
            linspace_x = np.linspace(resampled_index.min(), resampled_index.max(), len(resampled))
            energy_spline = interp1d(x=resampled_index, y = resampled.values)
            axes1.plot(resampled.index, energy_spline(linspace_x), label=f'{column} Linear Spline', lw=lw1, alpha=1, color='red')
        else:
            axes1.plot(df.index, df[column], label=column, lw=0.5, alpha=0.6, color=colours[colour_count])
            colour_count += 1
    axes1.set_xlabel('Time')
    axes1.set_ylabel(axes1_label)

    axes2 = axes1.twinx()
    axes2_label = labels2

    for column in axes2_dict[axes2_label]:
        if trendline_2:
            resampled = df[column].resample("5D").first()
            resampled_index = (resampled.index.tz_localize(None) - pd.Timestamp("1970-01-01")) /  pd.Timedelta(seconds=1)
            linspace_x = np.linspace(resampled_index.min(), resampled_index.max(), len(resampled))
            energy_spline = interp1d(x=resampled_index, y = resampled.values)
            axes2.plot(resampled.index, energy_spline(linspace_x), label=f'{column} Linear Spline', lw=lw2, alpha=1, color='black', ls="dotted")
        
            temp = add_seasons(df.copy())
            temp["Winter_or_Summer"] = np.where(((temp['season'] == 'Summer') | (temp['season'] == 'Winter')), 0 , np.nan)
            target_indexes = temp[~temp["Winter_or_Summer"].isna()].index

            axes1.set_ylim(axes1.get_ylim()[0] - 0.1 * (axes1.get_ylim()[1] - axes1.get_ylim()[0]), axes1.get_ylim()[1])
            axes1.scatter(df.loc[target_indexes].index, [axes1.get_ylim()[0] + 0.02 * (axes1.get_ylim()[1] - axes1.get_ylim()[0])] * len(target_indexes), 
                          color='red', label='Winter or Summer', marker='x', s=25, lw=0.75)
        else:
            axes2.plot(df.index, df[column], label=column, lw=0.5, alpha=0.5)
            colour_count += 1
    axes2.set_ylabel(axes2_label)

    fig.suptitle(title)
    lines, labels = axes1.get_legend_handles_labels()
    lines2, labels2 = axes2.get_legend_handles_labels()
    axes2.legend(lines + lines2, labels + labels2, loc='upper left')
    plt.tight_layout()
    plt.show()

# USED FOR WEATHER FEATURES
def plot_average_feature_by_city(df, column_list, y_lab, title):

    temp = df.copy()
    temp = add_seasons(temp)
    grouped_by_season = temp[["season"] + column_list].groupby(["season"]).mean().reset_index()
    grouped_by_season.set_index("season", inplace=True)
    grouped_by_season = grouped_by_season.reindex(["Winter", "Spring", "Summer", "Fall"])
    plot_linegraphs(grouped_by_season, grouped_by_season.columns, "Seasons", y_lab, title)

def plot_linegraphs(df, columns, xlab, ylab, title, lw=0.5, alpha=0.5, marker=''):

    plt.figure(figsize=(12, 6))
    for column in columns:
        plt.plot(df.index, df[column], label=column, lw=lw, alpha=alpha, marker=marker)
    plt.suptitle(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_linegraphs_target(df, columns, target, xlab, ylab, title, lw=0.5, alpha=0.5, marker=''):

    plt.figure(figsize=(12, 6))
    for column in columns:
        if column == target:
            plt.plot(df.index, df[col], label=column, lw=0.8, alpha=0.7)
        else:
            plt.plot(df.index, df[column], label=column, ls="--",lw=lw, alpha=alpha, marker=marker)
    plt.suptitle(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_linegraphs_target(df, columns, target, xlab, ylab, title, lw=1.2, alpha=1, marker='', target_color='black', colours=[]):

    if colours == []:
        colours = get_n_colours(len(columns)-1, [target_color])
        
    plt.figure(figsize=(18, 6))
    count = 0
    for column in columns:
        if column == target:
            plt.plot(df.index, df[column], label=column, lw=1, alpha=1, color=target_color)
        else:
            if count % 2 == 0:
                plt.plot(df.index, df[column], label=column, ls="-.", lw=lw, alpha=alpha, marker=marker, color=colours[count])
            else:
                plt.plot(df.index, df[column], label=column, ls=":", lw=lw, alpha=alpha, marker=marker, color=colours[count])
            count += 1
    plt.suptitle(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_decompose_ts(df, model, period=24):
    
    if model in ('additive', 'multiplicative'):

        temp = df.copy()
        decomp = decompose_ts(temp, model, period)
        fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        temp.plot(ax=axes[0])
        decomp.trend.plot(ax=axes[1])
        axes[1].set_title("Trend")
        decomp.seasonal.plot(ax=axes[2])
        axes[2].set_title("Seasonal")
        decomp.resid.plot(ax=axes[3])
        axes[3].set_title("Resid")
        
        plt.tight_layout()
        plt.show()
    else:
        print("Invalid model type. Please specify 'additive' or 'multiplicative'.")

def plot_acf(df, target, nlags=170, i_confint=False):
    
    temp = df.copy()[target].dropna()
    acf_values, confint = acf(temp, nlags=nlags, alpha=0.05)

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(acf_values)), acf_values, marker=".", linestyle="-", color="red")
    if i_confint:
        confint_lower = confint[:, 0] - acf_values
        confint_upper = confint[:, 1] - acf_values
        plt.fill_between(range(len(acf_values)), confint_lower, confint_upper, color='lightgrey', alpha=0.7) 
    bound = np.ceil(nlags/24).astype(int)
    if bound > 1:
        plt.vlines(x=[24*i for i in range(1,bound)], ymin=min(acf_values), ymax=max(acf_values), lw=0.5, ls= "--", color="blue", alpha=0.6)
    plt.title('Autocorrelation Function (ACF)')
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelation')
    plt.show()

def plot_pacf(df, target, nlags=170, i_confint=False):
    
    temp = df.copy()[target].dropna()
    pacf_values, confint = pacf(temp, nlags=nlags, alpha=0.05)
   
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(pacf_values)), pacf_values, marker=".", linestyle="-", color="red")
    if i_confint:
        confint_lower = confint[:, 0] - pacf_values
        confint_upper = confint[:, 1] - pacf_values
        plt.fill_between(range(len(pacf_values)), confint_lower, confint_upper, color='lightgrey', alpha=0.7) 
    bound = np.ceil(nlags/24).astype(int)
    if bound > 1:
        plt.vlines(x=[24*i for i in range(1,bound)], ymin=min(pacf_values), ymax=max(pacf_values), lw=0.5, ls= "--", color="blue", alpha=0.6)
    plt.title('Partial Autocorrelation Function (PACF)')
    plt.xlabel('Lags')
    plt.ylabel('Partial Autocorrelation')
    plt.show()

def plot_cff(df, target1, target2):
    
    series1 = df[target1].copy()
    series2 = df[target2].copy()
    ccf_values = ccf(series1, series2)
    
    plt.stem(ccf_values)
    plt.title(f'Cross-Correlation between {target1} and {target2}')
    plt.xlabel('Lags')
    plt.ylabel('Cross-Correlation')
    plt.show()

def plot_corr_comp(df1, df2, title_1="", title_2 ="", suptitle=""):

    fig, axes = plt.subplots(1,2, figsize=(28,12))
    
    corr_matrix = df1.corr()
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0 , ax=axes[0])
    axes[0].set_title(title_1, fontsize=15)
    axes[0].tick_params(labelsize=12)

    corr_matrix_2 = df2.corr()
    sns.heatmap(corr_matrix_2, annot=False, cmap="coolwarm", center=0, ax=axes[1])
    axes[1].set_title(title_2, fontsize=15)
    axes[1].tick_params(labelsize=12)

    plt.suptitle(suptitle, fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98]) 
    plt.show()

def plot_scree(pca_object):

    explained_variance = pca_object.explained_variance_ratio_
    num_components = len(pca_object.explained_variance_ratio_)

    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(1, num_components + 1), explained_variance, alpha=0.7, label='Individual component variance')
    plt.plot(np.arange(1, num_components + 1), np.cumsum(explained_variance), marker='o', linestyle='--', color='r', label='Cumulative variance')
    plt.xlabel('Principal components')
    plt.ylabel('Explained variance ratio')
    plt.title(f'Explained Variance for {num_components} Components')
    plt.xticks(np.arange(1, len(explained_variance) + 1, step=5))
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_model_rmse(history, path=""):

    train_rmse = history.history['root_mean_squared_error']
    valid_rmse = history.history['val_root_mean_squared_error']

    rmse_df = pd.DataFrame({"Train RMSE": train_rmse, "Validation RMSE": valid_rmse})
    if path != "":
        save_file(rmse_df, path, False)

    plt.figure(figsize=(12, 6))
    plt.plot(train_rmse, label="Training RMSE", color="Blue")
    plt.plot(valid_rmse, label="Validation RMSE", color="Red")
    plt.title("Epochs vs. Training and Validation RMSE")
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    plt.show()

#####################################################################################
# Imputation Helpers #
#####################################################################################

def impute_previous_n_days(df, days_back, ignore_holiday=False):

    missing_index = df.loc[df.isna()].index

    holiday_list = []
    if not ignore_holiday:
        temp = pd.DataFrame(df.copy())
        temp = is_holidays(temp, temp.index.year.unique())
        holiday_list = list(set(temp[temp["isHoliday"] == 1].index.date))
        
    for idx in missing_index:
        found_value = False
        for day in range(days_back, 0, -1):
            temp_idx = idx - pd.Timedelta(days=day)

            if temp_idx.date() in holiday_list:
                continue
            # Check if previous date to grab value from exists
            # Check if previous date to grab value from is not missing as well
            if temp_idx in df.index and not pd.isna(df.loc[temp_idx]):
                df.loc[idx] = df.loc[temp_idx]
                found_value = True
                break
        if not found_value:
            print(f"Some values were unable to impute! - from {df.name}")
            df.loc[idx] = np.nan
        
    return df

def impute_stat_n_days(df, days_back, func, ignore_holiday=False):

    missing_index = df.loc[df.isna()].index

    holiday_list = []
    if not ignore_holiday:
        temp = pd.DataFrame(df.copy())
        temp = is_holidays(temp, temp.index.year.unique())
        holiday_list = list(set(temp[temp["isHoliday"] == 1].index.date))

    
    for idx in missing_index:
        found_value = False
        values = []
        for day in range(days_back+1):
            temp_idx = idx - pd.Timedelta(days=day)
            if temp_idx.date() in holiday_list:
                continue
            if temp_idx in df.index and not pd.isna(df.loc[temp_idx]):
                values.append(df.loc[temp_idx])
                found_value = True # one value is enough 
            
        if not found_value:
            print(f"Some values were unable to impute! - from {df.name}")
            df.loc[idx] = np.nan
        else:
            df.loc[idx] = func(values)
            
    return df

def custom_ffill_n_hours(df, hours_back, ignore_holiday=False):

    missing_index = df.loc[df.isna()].index

    holiday_list = []
    if not ignore_holiday:
        temp = pd.DataFrame(df.copy())
        temp = is_holidays(temp, temp.index.year.unique())
        holiday_list = list(set(temp[temp["isHoliday"] == 1].index.date))
          
    for idx in missing_index:
        found_value = False
        for day in range(hours_back+1):
            temp_idx = idx - pd.Timedelta(hours=hours_back)
            if temp_idx.date() in holiday_list:
                continue
            if temp_idx in df.index and not pd.isna(df.loc[temp_idx]):
                df.loc[idx] = df.loc[temp_idx]
                found_value = True
                break

        if not found_value:
            df.loc[idx] = custom_ffill_n_hours(df, hours_back+1)
    
    if df.isna().any():
        print(f"Some values were unable to impute! - from {df.name}")

    return df

#####################################################################################
# Features Generation #
#####################################################################################

def is_weekday(df):

    df["isWeekday"] = (df.index.dayofweek < 5).astype(float) 

    return df

def is_holidays(df, years, code="ES"):

    holidates = holiday_dates(years, code)
    df["isHoliday"] = [1 if dte in holidates else 0 for dte in df.index.date]
    
    return df

def add_seasons(df, feature_gen = False):

    season_dict = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring',
               5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer',
               9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'}
    
    # Cyclical Encoding for seasons
    if feature_gen:
        
        encode_dict = {"Winter": 1, "Spring": 2, "Summer": 3, "Fall": 4}

        seasons = pd.Series(df.index.month).apply(lambda x: season_dict[x])
        seasons_encode = seasons.apply(lambda y: encode_dict[y])
        df["season_sin"] = (np.sin(2 * np.pi * seasons_encode.values / 4) + 1) / 2
        df["season_cos"] = (np.cos(2 * np.pi * seasons_encode.values / 4) + 1) / 2
        return df

    else:
    
        if "month" in df.columns:
            df["season"] = df["month"].apply(lambda x: season_dict[x])
            return df
        else: 
            df["month"] = df.index.month
            df["season"] = df["month"].apply(lambda x: season_dict[x])
            df.drop(["month"], axis=1, inplace=True)
            return df

def add_time_features(df, feature_gen = False):
    
    # Cyclical Encoding for each time-based feature
    if feature_gen:

        hour = df.index.hour
        month = df.index.month
        dayofweek = df.index.dayofweek # Monday == 0 and Sunday == 6
        weekofyear = df.index.isocalendar().week

        df["hour_sin"] = (np.sin(2 * np.pi * hour / 24) + 1) / 2
        df["hour_cos"] = (np.cos(2 * np.pi * hour / 24) + 1) / 2

        df["month_sin'"] = (np.sin(2 * np.pi * month / 12) + 1) / 2
        df["month_cos"] = (np.cos(2 * np.pi * month / 12) + 1) / 2

    else:
        # For data analysis
        df['hour'] = df.index.day
        df['hour'] = df.index.hour
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['dayofweek'] = df.index.dayofweek # Monday == 0 and Sunday == 6
        df['weekofyear'] = df.index.isocalendar().week

    return df

def add_business_hours(df, feature_gen = False):

    indicators = []
    
    for hour in df.index.hour:
        # business hours
        if (hour > 8 and hour < 14) or (hour > 16 and hour < 20): 
            indicators.append(1)
        # Siesta
        elif (hour >= 14 and hour <= 16):
            indicators.append(2)
        # After business hours
        else:
            indicators.append(0)
    df["is_business_hour"] = np.array(indicators)

    if feature_gen:

        encoded_df = pd.get_dummies(df["is_business_hour"]).astype(float)
        encoded_df.columns = ["is_after_business_hours", "is_business_hours", "is_siesta"]

        df = pd.concat([df, encoded_df], axis=1)
        df.drop('is_business_hour', axis=1, inplace=True)
        return df

    return df

def add_lags_by_hours(df, target, lags=[0]):

    series = df[target].copy()
    for lag in lags:
        df[f"lag_{lag} {target}"] = series.shift(lag)

    return df

def add_temp_ranges(df, cities=None):

    # Case when weather features have not been merged in yet
    if "temp" in df.columns or cities == None:
        print("Note: City data has not been added.")
        df["temp_range"] = df["temp_max"] - df["temp_min"]
        return df
    else:
        for city in cities:
            df[f"temp_range_{city}"] = np.abs(df[f"temp_max_{city}"] - df[f"temp_min_{city}"])
        return df
    
def add_interaction_pairs(df, feature1_prefix, feature2_prefix,  cities):

    for city in cities:
        feature1, feature2 = f"{feature1_prefix}_{city}", f"{feature2_prefix}_{city}"
        df[f"{feature1_prefix}_x_{feature2_prefix}_{city}"] = df[feature1] * df[feature2]

    return df

def is_anomaly_by_season_hour(df, targets, groupby_list=["season", "hour"]):

    for col in targets:
        temp = add_seasons(df.copy())
        temp = add_time_features(temp)
        temp = temp[[col] + groupby_list].copy()
        stats = wf_groupby_stats(temp, col, groupby_list)
        
        for season_hour, data in temp.groupby(groupby_list):
            season, hour = season_hour[0], season_hour[1]
            skewness_by_season_hour = stats[(stats["season"] == season) & (stats["hour"] == hour)]["skewness"]
            skewness = skewness_by_season_hour[skewness_by_season_hour.index[0]]

            if abs(skewness) < 0.5: # if symmetrical
                anomalies = detect_anomalies_zscore(data[col])
            else:
                anomalies = detect_anomalies_percentile(data[col])

            df.loc[data.index, f"{col}_anomalies"] = anomalies

    return df  

#####################################################################################
# Model Evaluation (Loss Functions) & Data Transformation # 
#####################################################################################


def evaluate_model(actual, predicted, train=True, index = None):

    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    if train:    
        metrics_df = pd.DataFrame({"MSE": [mse], 
                                   "RMSE": [rmse]})
    else:
        mae = np.mean(np.abs(actual - predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        metrics_df = pd.DataFrame({"MSE": [mse], 
                                "RMSE": [rmse],
                                "MAE": [mae],
                                "MAPE": [mape]})
    if index is not None:
        metrics_df.index=index

    return metrics_df

def demand_stationary(raw_data, columns):

    temp = raw_data[columns].copy()
    shifted_data = 1 + temp + np.abs(np.min(temp, axis=0))
    log_data = np.log(shifted_data)
    fd_data = log_data.diff(1)

    return fd_data

def reverse_stationary(stationary_data, raw_data, columns):

    station_temp = stationary_data[columns].copy()
    raw_temp = raw_data[columns].copy()
    abs_min = np.abs(np.min(raw_temp, axis=0))
    shifted_data = np.log(1 + raw_temp.shift(1) + abs_min)

    series = station_temp + shifted_data
    series.dropna(inplace=True)
    series = np.exp(series) - 1 - abs_min
    
    index_date = series.index[0]
    combine_raw = raw_temp[(raw_temp.index < index_date)]
    final_df = pd.concat([combine_raw, series]).sort_index()

    return final_df

def get_min_max_features(df, axis=0):

    min_features = np.min(df, axis=axis)
    max_features = np.max(df, axis=axis)

    return min_features, max_features

def min_max_scaler(data, feature_range=(0, 1), find_min_max=False, min_features=None, max_features=None):

    min_val, max_val = feature_range
    temp = data.copy()
    if find_min_max:
        min_features, max_features = get_min_max_features(temp)
        denominator = max_features - min_features
        standard = (temp - min_features) / np.where(denominator==0, 1, denominator)
        scaled = standard * (max_val - min_val) + min_val
        return scaled, min_features, max_features
    else:
        if min_features is None or max_features is None: # Need min or max features for validation or testing data
            raise ValueError("Need to provide both min_features or max_features to proceed.")
        else:
            denominator = max_features - min_features
            standard = (temp - min_features) / np.where(denominator==0, 1, denominator)
            scaled = standard * (max_val - min_val) + min_val
            return scaled

def reverse_min_max_scaler(scaled_data, min_features, max_features, feature_range=(0, 1)):

    min_val, max_val = feature_range
    standard = (scaled_data - min_val) / (max_val - min_val)
    original = (standard * (max_features - min_features)) + min_features

    return original

def df_stationary(df, numerical_columns, qualitative_columns):

    df.sort_index(inplace=True)
    numerical_df = demand_stationary(df, numerical_columns)
    numerical_df.dropna(axis=0, inplace=True)
    qualitative_df = df[qualitative_columns].copy()
    stationary_df = numerical_df.merge(qualitative_df, left_index=True, right_index=True, how="inner")
    sorted_columns = maintain_feature_order(df.columns.tolist(), stationary_df.columns.tolist())

    return stationary_df[sorted_columns]

#####################################################################################
# Feature Selection #
#####################################################################################

def train_test_split(df_to_split, raw_df_index, test_size=0.3):

    train_index = raw_df_index[np.floor(len(raw_df_index) * (1-test_size)).astype(int)]
    train_data = df_to_split[df_to_split.index < train_index].copy()
    test_data = df_to_split[df_to_split.index >= train_index].copy()

    return train_data, test_data

def split_train_test_validation(df_to_split, raw_df_index, test_size=0.15, validation_size=0.15):
    
    train_index = raw_df_index[np.floor(len(raw_df_index) * (1-(test_size + validation_size))).astype(int)]
    test_index = raw_df_index[np.floor(len(raw_df_index) * (1-(test_size))).astype(int)]
    
    train_data = df_to_split[df_to_split.index < train_index].copy()
    validation_data = df_to_split[(df_to_split.index >= train_index) & (df_to_split.index < test_index)].copy()
    test_data = df_to_split[df_to_split.index >= test_index].copy()

    return train_data, validation_data, test_data

def tss_splits(split_options):

    split_options = list(itertools.product(split_options['n_splits'], split_options['test_size'], split_options['gap']))    
    split_options.sort(key=lambda x: x[0])

    return split_options

################################### XGBRegressor ####################################

def xgb_feature_importance(train_data, test_data, target, param_grid, tss_params):
    
    X_train, X_test = train_data.drop(columns=target).copy(), test_data.drop(columns=target).copy()
    y_train, y_test = train_data[target].copy(), test_data[target].copy()

    n_splits, test_size, gap = tss_params["n_splits"], tss_params["test_size"], tss_params["gap"]
    tss = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)

    xgb = XGBRegressor(random_state=21)
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=tss, scoring='neg_mean_squared_error', n_jobs=-1)

    grid_search.fit(X_train, y_train)
    best_n_estimators = grid_search.best_params_['n_estimators']
    best_max_depth = int(grid_search.best_params_['max_depth'])
    best_learning_rate = grid_search.best_params_['learning_rate']

    cv_results = grid_search.cv_results_
    mean_cv_mse = -cv_results['mean_test_score'][grid_search.best_index_]
    mean_cv_rmse = np.sqrt(mean_cv_mse)
    rf_best_cv_results = pd.DataFrame([{'MSE': mean_cv_mse, 'RMSE': mean_cv_rmse}])
    
    xgb_best = XGBRegressor(n_estimators=best_n_estimators, max_depth=best_max_depth, learning_rate=best_learning_rate, random_state=21)
    xgb_best.fit(X_train, y_train)
    
    feature_importances_best = xgb_best.feature_importances_
    feature_importance_df_best = pd.DataFrame({'Feature': X_train.columns,
                                               'Importance': feature_importances_best}).sort_values(by='Importance', ascending=False)

    y_pred = xgb_best.predict(X_test)
    xgb_best_results = evaluate_model(y_test, y_pred)

    plt.figure(figsize=(8, 10))
    plt.barh(np.arange(len(feature_importance_df_best['Feature'])), feature_importance_df_best['Importance'], color='skyblue', align='center')
    plt.yticks(np.arange(len(feature_importance_df_best['Feature'])), feature_importance_df_best['Feature'], fontsize=8)
    plt.gca().invert_yaxis()
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.title(f'Feature Importance from XGBRegressor (n_estimators={best_n_estimators})', fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(left=0.3)
    plt.grid(True)
    plt.show()

    print("---------- XGBoost Cross Validation ---------- \n")
    print(f"Best n_estimators from CV: {best_n_estimators} \n")
    print(f"Best max_depth from CV: {best_max_depth} \n")
    print(f"Best learning_rate from CV: {best_learning_rate} \n")
    print("XGBRegressor CV Model Loss Metrics: \n")
    print(xgb_best_results)

    return feature_importance_df_best, best_n_estimators, int(best_max_depth), best_learning_rate, xgb_best_results

def xgb_evaluate_model(train_data, test_data, target, n_estimators, max_depth, learning_rate, features):

    train_data = train_data[features].copy()
    test_data = test_data[features].copy()

    X_train, X_test = train_data.drop(columns=target), test_data.drop(columns=target)
    y_train, y_test = train_data[target], test_data[target]

    xgb = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=21)
    xgb.fit(X_train, y_train)   
    y_pred = xgb.predict(X_test)

    results = evaluate_model(y_test, y_pred)

    print("------------ XGBoost Final Results ------------ \n")
    print(f"Selected Important Features: {features} \n")
    print("XGBoost Final Loss Metrics: \n")
    print(results)

    return results, features

################################### Random Forest ###################################

def rf_feature_importance(train_data, test_data, target, param_grid, tss_params):

    X_train, X_test = train_data.drop(columns=target).copy(), test_data.drop(columns=target).copy()
    y_train, y_test = train_data[target].copy(), test_data[target].copy()

    n_splits, test_size, gap = tss_params["n_splits"], tss_params["test_size"], tss_params["gap"]
    tss = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)

    rf = RandomForestRegressor(random_state=21)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=tss, scoring='neg_mean_squared_error', n_jobs=-1)

    grid_search.fit(X_train, y_train)
    best_n_estimators = grid_search.best_params_['n_estimators']
    best_max_depth = grid_search.best_params_['max_depth']

    rf_best = RandomForestRegressor(n_estimators=best_n_estimators, max_depth=best_max_depth, random_state=21)
    rf_best.fit(X_train, y_train)
    feature_importances_best = rf_best.feature_importances_
    feature_importance_df_best = pd.DataFrame({'Feature': X_train.columns,
                                               'Importance': feature_importances_best}).sort_values(by='Importance', ascending=False)

    y_pred = rf_best.predict(X_test)
    rf_best_results = evaluate_model(y_test, y_pred)

    plt.figure(figsize=(8, 10))
    plt.barh(np.arange(len(feature_importance_df_best['Feature'])), feature_importance_df_best['Importance'], color='skyblue', align='center')
    plt.yticks(np.arange(len(feature_importance_df_best['Feature'])), feature_importance_df_best['Feature'], fontsize=8)
    plt.gca().invert_yaxis()

    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.title(f'Feature Importance from Random Forest Regressor (n_estimators={10})', fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(left=0.3)
    plt.grid(True)
    plt.show()

    print("---------- Random Forest Cross Validation ---------- \n")
    print(f"Best n_estimators from CV: {best_n_estimators} \n")
    print(f"Best max_depth from CV: {best_max_depth} \n")
    print("Random Forest Cross Validation Loss Metrics: \n")
    print(rf_best_results)
    
    return feature_importance_df_best, best_n_estimators, best_max_depth, rf_best_results

def rf_evaluate_model(train_data, test_data, target, n_estimators, features):

    train_data = train_data[features].copy()
    test_data = test_data[features].copy()

    X_train, X_test = train_data.drop(columns=target), test_data.drop(columns=target)
    y_train, y_test = train_data[target], test_data[target]

    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=best_max_depth, random_state=21)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    results = evaluate_model(y_test, y_pred)

    print("------------ Random Forest Final Results ------------ \n")
    print("Random Forest Final Loss Metrics: \n")
    print(results)

    return results, features

####################################### Lasso #######################################

def lasso_feature_selection(train_data, test_data, target, tss_params, ref_features, alphas=np.logspace(-4, 0, 50)):

    X_features = ref_features.copy()
    X_features.remove(target)
    features = maintain_feature_order(ref_features, X_features)
    
    min_val, max_val = get_min_max_features(train_data)
    train_data = min_max_scaler(train_data, min_features=min_val, max_features=max_val)
    test_data = min_max_scaler(test_data, min_features=min_val, max_features=max_val)

    X_train, X_test = train_data.drop(columns=target), test_data.drop(columns=target)
    y_train, y_test = train_data[target], test_data[target]
    X_train, X_test = X_train[features], X_test[features] 

    n_splits, test_size, gap = tss_params["n_splits"], tss_params["test_size"], tss_params["gap"]
    tss = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)

    pipeline = Pipeline([('scaler', MinMaxScaler()), 
                         ('lasso_cv', LassoCV(alphas=alphas, cv=tss, random_state=21, n_jobs=-1))])
    pipeline.fit(X_train, y_train)
    
    best_alpha = pipeline.named_steps['lasso_cv'].alpha_
    mse_cv = np.mean(pipeline.named_steps['lasso_cv'].mse_path_, axis=1).min()
    rmse_cv = np.sqrt(mse_cv)
    lasso_best_alpha_results = pd.DataFrame([{'MSE': mse_cv, 'RMSE': rmse_cv}])

    lasso_cv_model = pipeline.named_steps['lasso_cv']
    lasso_coefficients = lasso_cv_model.coef_
    selected_features_indices = np.where(lasso_coefficients != 0)[0]
    selected_feature_names = [features[i] for i in selected_features_indices]

    print("---------- Lasso Cross Validation ---------- \n")
    print(f"Best alpha: {best_alpha.round(5)} \n")
    print(f"Selected features: {selected_feature_names} \n")
    print(f"*{len(selected_feature_names)} features selected.* \n")
    print("Lasso CV Model Loss Metrics: \n")
    print(lasso_best_alpha_results)
    
    optimal_lasso = Lasso(alpha=best_alpha, random_state=21)
    optimal_lasso.fit(X_train[selected_feature_names], y_train)

    y_pred = optimal_lasso.predict(X_test[selected_feature_names])
    metrics_df = evaluate_model(y_test, y_pred)

    lasso_coefficients = optimal_lasso.coef_
    selected_features_indices = np.where(lasso_coefficients != 0)[0]
    selected_feature_names_final = [selected_feature_names[i] for i in selected_features_indices]

    print("---------- Final Features Selected ---------- \n")
    print(f"Selected features: {selected_feature_names} \n")
    print(f"*{len(selected_feature_names)} features selected.* \n")
    print("Lasso Final Model Loss Metrics: \n")
    print(metrics_df)

    return selected_feature_names_final

def evaluate_optimal_lasso(train_data, test_data, target, alpha, selected_features, ref_features):
    
    features = maintain_feature_order(ref_features,  [target] + selected_features)
 
    min_val, max_val = get_min_max_features(train_data)
    train_data = min_max_scaler(train_data, min_features=min_val, max_features=max_val)
    test_data = min_max_scaler(test_data, min_features=min_val, max_features=max_val)
    train_data, test_data = train_data[features], test_data[features]

    X_train, X_test = train_data.drop(columns=target), test_data.drop(columns=target)
    y_train, y_test = train_data[target], test_data[target]

    optimal_lasso = Lasso(alpha=alpha, random_state=21)
    optimal_lasso.fit(X_train, y_train)

    y_pred = optimal_lasso.predict(X_test)
    metrics_df = evaluate_model(y_test, y_pred)

    lasso_coefficients = optimal_lasso.coef_
    selected_features_indices = np.where(lasso_coefficients != 0)[0]
    selected_feature_names = [features[i] for i in selected_features_indices]

    print("---------- Final Features Selected ---------- \n")
    print(f"Selected features: {selected_feature_names} \n")
    print(f"*{len(selected_feature_names)} features selected.* \n")
    print("Lasso Final Model Loss Metrics:")
    print(metrics_df)

    return selected_feature_names, metrics_df

def fine_tune_alpha(train_data, target, tss_params, alphas=np.logspace(-5, 1, 50)):

    X = train_data.drop(columns=target).copy()
    y = train_data[target].copy()
    n_splits, test_size, gap = tss_params["n_splits"], tss_params["test_size"], tss_params["gap"]
    tss = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)

    pipeline = Pipeline([('scaler', MinMaxScaler()), 
                         ('lasso_cv', LassoCV(alphas=alphas, cv=tss, random_state=21, n_jobs=-1))])
    pipeline.fit(X, y)
    
    best_alpha = pipeline.named_steps['lasso_cv'].alpha_
    mse_cv = np.mean(pipeline.named_steps['lasso_cv'].mse_path_, axis=1).min()
    rmse_cv = np.sqrt(mse_cv)

    results = pd.DataFrame([{'MSE': mse_cv, 'RMSE': rmse_cv}])

    features = train_data.columns.tolist()
    lasso_coefficients = pipeline.named_steps['lasso_cv'].coef_
    selected_features_indices = np.where(lasso_coefficients != 0)[0]
    selected_feature_names = [features[i] for i in selected_features_indices]

    print(f"Best alpha: {best_alpha} \n")
    print(f"Selected features: {selected_feature_names}")
    print(f"*{len(selected_feature_names)} features selected.* \n")

    return best_alpha, selected_feature_names, results

def lasso_feature_selection_v1(df, target, tss_params, alpha_grid):
        
    print("---------- Lasso Cross Validation ---------- \n")
    best_alpha, selected_features, results_cv = fine_tune_alpha(df, target, tss_params, alpha_grid)
    print("Lasso Cross Validation Loss Metrics:")
    print(results_cv , "\n")

    return selected_features, best_alpha, results_cv

######################################## PCA ########################################

def initial_PCA_scree(df, target):

    X_train = df.drop(columns=target).copy() 
    X_train, _, _ = min_max_scaler(X_train, find_min_max=True)
    pca = PCA(random_state=21)
    pca.fit(X_train)
    plot_scree(pca)

def apply_PCA(data, target, n_components):

    df = data.drop(columns=target).copy()
    features_order = maintain_feature_order(data.columns.to_list(), df.columns.tolist())
    df, _, _ = min_max_scaler(df, find_min_max=True)
    df = df[features_order]
    
    pca = PCA(n_components=n_components, random_state=21)
    pca.fit(df) 
    variance_explained = np.sum(pca.explained_variance_ratio_)
    pca_data = pca.transform(df)
    pca_data = pd.DataFrame(pca_data, columns=[f"PC_{i+1}" for i in range(len(pca_data[0]))])
    pca_data.index = data.index

    print(f"The number of components is {len(pca_data.columns)}. \n")
    print(f"The variance explained is {variance_explained}. \n")

    return pca_data

def pca_feature_selection(data, target, n_components):
    
    X = data.drop(columns=target).copy() 
    y = data[target].copy()

    X_scaled, _ , _ = min_max_scaler(X, find_min_max=True)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    X_pca = pd.DataFrame(X_pca, columns=[f"PC_{i+1}" for i in range(len(X_pca[0]))])
    X_pca.index = X.index
    pca_df = pd.merge(X_pca, y, left_index=True, right_index=True, how='outer')
    pca_df = pca_df.sort_index()

    variance_explained = np.sum(pca.explained_variance_ratio_)
    print(f"Reducing {len(X.columns)} features to {n_components} components results in a total explained variance of {(variance_explained*100).round(2)}%. \n")

    return pca_df

def fine_tune_pca(df, target, split_options, param_grid):

    X = df.drop(columns=target).copy() 
    y = df[target].copy()

    all_results = []

    for n_splits, test_size, gap in split_options:
        tss = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)

        pipeline = Pipeline([('scaler', MinMaxScaler()),
                             ('pca', PCA(random_state=21)),    
                             ('xgb', XGBRegressor(random_state=21))])         
        
        grid_search = GridSearchCV(pipeline, param_grid, cv=tss, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X, y)

        best_pca = grid_search.best_estimator_.named_steps['pca']
        cumulative_explained_variance = np.sum(best_pca.explained_variance_ratio_)
        mse_cv = np.mean(-1 * grid_search.cv_results_['mean_test_score'])
        rmse_cv = np.sqrt(mse_cv) 

        all_results.append({
            'n_splits': n_splits,
            'test_size': test_size,
            'gap': gap,
            'best_n_components': grid_search.best_params_['pca__n_components'],
            'explained_variance': cumulative_explained_variance,
            'best_learning_rate': grid_search.best_params_['xgb__learning_rate'],
            'best_n_estimators': grid_search.best_params_['xgb__n_estimators'],
            'best_max_depth': grid_search.best_params_['xgb__max_depth'],
            'best_alpha': grid_search.best_params_['xgb__alpha'],
            'best_lambda': grid_search.best_params_['xgb__lambda'],
            'mse': mse_cv,
            'rmse': rmse_cv})
        
    results_df = pd.DataFrame(all_results)
    results_df.sort_values(by="rmse", inplace=True)
    best_pca_config = results_df.iloc[0]
    
    return best_pca_config

def pca_feature_selection_v1(df, target, split_options, param_grid):

    split_combinations = tss_splits(split_options)
    best_pca_config =  fine_tune_pca(df, target, split_combinations, param_grid)

    n_components = best_pca_config['best_n_components']
    var_explained = best_pca_config['explained_variance']
    learning_rate = best_pca_config['best_learning_rate']
    n_estimators = best_pca_config['best_n_estimators']
    max_depth = best_pca_config['best_max_depth']
    alpha = best_pca_config['best_alpha']
    lambda_ = best_pca_config['best_lambda']
    mse = best_pca_config['mse']
    rmse = best_pca_config['rmse']
    cv_loss_results = pd.DataFrame([{"MSE": mse, "RMSE": rmse}])    

    print("---------- PCA Cross Validation ---------- \n")
    print(f"The number of components selected is {n_components}. \n")
    print(f"The variance explained is {var_explained}. \n")
    print("PCA Cross Validation Loss Metrics:")
    print(cv_loss_results, "\n")

    print("Optimal Parameters: \n")
    print(f"Optimal Learning Rate: {learning_rate}")
    print(f"Optimal Max Depth: {max_depth.astype(int)}")
    print(f"Optimal Alpha: {alpha}")
    print(f"Optimal Alpha: {lambda_}")
    
    return n_components.astype(int), learning_rate, n_estimators.astype(int), max_depth.astype(int), alpha, lambda_

def refit_and_plot_pca(X_train, best_n_components, target):

    pca = PCA(n_components=best_n_components, random_state=21)
    pca.fit(X_train)
    plot_scree(pca)

    return pca

def evaluate_optimal_pca(train_data, test_data, target, n_components, learning_rate, n_estimators, max_depth, alpha, lambda_, ref_features):

    features_order = maintain_feature_order(ref_features, train_data.columns.tolist())
    min_val, max_val = get_min_max_features(train_data)

    train_data = min_max_scaler(train_data, min_features=min_val, max_features=max_val)
    test_data = min_max_scaler(test_data, min_features=min_val, max_features=max_val)
    train_data, test_data = train_data[features_order], test_data[features_order]
    X_train, X_test = train_data.drop(columns=target), test_data.drop(columns=target) 
    y_train, y_test = train_data[target], test_data[target]

    # refit and plot PCA
    pca = PCA(n_components=n_components, random_state=21)
    pca.fit(X_train)
    plot_scree(pca)

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    optimal_xgb = XGBRegressor(learning_rate=learning_rate, 
                               n_estimators=n_estimators, 
                               max_depth=max_depth,
                               alpha=alpha,
                               lambda_=lambda_,
                               random_state=21)
    optimal_xgb.fit(X_train_pca, y_train)

    y_pred = optimal_xgb.predict(X_test_pca)
    metrics_df = evaluate_model(y_test, y_pred)


    non_zero_indices = np.where(optimal_xgb.feature_importances_ != 0)[0]
    remaining_variance_explained = pca.explained_variance_ratio_[non_zero_indices]
    remaining_components = len(non_zero_indices)
    total_remaing_variance = np.sum(remaining_variance_explained)

    X_train_pca = pd.DataFrame(X_train_pca[:, non_zero_indices], columns=[f"PC_{i+1}" for i in range(len(X_train_pca[:, non_zero_indices][0]))])
    X_test_pca = pd.DataFrame(X_test_pca[:, non_zero_indices], columns=[f"PC_{i+1}" for i in range(len(X_test_pca[:, non_zero_indices][0]))])
    X_train_pca.index = train_data.index
    X_test_pca.index = test_data.index

    X_pca = pd.concat([X_train_pca, X_test_pca])
    y = pd.concat([y_train, y_test])
    pca_df = pd.merge(X_pca, y, left_index=True, right_index=True, how='outer')
    pca_df = pca_df.sort_index()
    
    print("----------- PCA Final Results ------------ \n")
    print(f"The number of components remaining is {len(X_pca.columns)}. \n")
    print(f"The variance explained is {total_remaing_variance}. \n")
    print("PCA Final Model Loss Metrics:")
    print(metrics_df)

    return pca_df

#####################################################################################
# Model Building #
#####################################################################################

def set_tf_seed():

    random.seed(21)
    np.random.seed(21)
    tf.random.set_seed(21)
    tf.config.experimental.enable_op_determinism()

def model_data_process(df, target, train_date_end, X_features, n_components):

    data = df.copy().dropna()
    train_data = data[data.index <= train_date_end]
    test_data = data[data.index > train_date_end]
    features_order = maintain_feature_order(X_features, train_data.columns.tolist())
   
    min_val, max_val = get_min_max_features(train_data)
    train_data = min_max_scaler(train_data, min_features=min_val, max_features=max_val)
    test_data = min_max_scaler(test_data, min_features=min_val, max_features=max_val)
    X_train, X_test = train_data[features_order], test_data[features_order]
    y_train, y_test = train_data[target], test_data[target]
    X_scaled = pd.concat([X_train, X_test])
    y_scaled = pd.concat([y_train, y_test])
    
    pca = PCA(n_components=n_components, random_state=21)
    pca.fit(X_train)
    X_pca = pca.transform(X_scaled[features_order])
    y_scaled = y_scaled.values.reshape(-1,1)

    processed_df = np.concatenate((X_pca, y_scaled), axis=1)
    return processed_df, train_data.index.union(test_data.index), min_val, max_val

def DNN_create_sliding_windows(processed_data, processed_index, start_date, end_date, predict_window, hourly_steps):

    set_tf_seed()

    X, y = [], []
    target_feature = processed_data[:,-1]

    if start_date not in processed_index:
        start_date = processed_index[0]
    
    start_date = start_date + pd.Timedelta(hours=predict_window) 
    start_index, end_index = processed_index.get_loc(start_date), processed_index.get_loc(end_date)
    for dte_index in range(start_index, end_index + 1):
        rng_X = range(dte_index - predict_window, dte_index, hourly_steps)
        current_X = processed_data[rng_X]
        current_y = target_feature[dte_index]
        X.append(current_X)
        y.append(current_y)
        
    return np.array(X), np.array(y)

def DNN_shuffle(train_tuple, valid_tuple, batch_size=32, buffer_size=1000):

    set_tf_seed()

    train = Dataset.from_tensor_slices(train_tuple).cache()
    train = train.shuffle(buffer_size=buffer_size, seed=21).batch(batch_size=batch_size).prefetch(5)

    valid = Dataset.from_tensor_slices(valid_tuple).cache()
    valid = valid.batch(batch_size=batch_size).prefetch(5)
    
    return train, valid

def DNN_predict(X_test, y_test, target, min_val, max_val, path):

    set_tf_seed()

    best_model = load_model(path)
    best_model.summary()

    y_pred = best_model.predict(X_test)

    y_pred_inv = reverse_min_max_scaler(y_pred.flatten(), min_val[target], max_val[target])
    y_test_inv = reverse_min_max_scaler(y_test.flatten(), min_val[target], max_val[target])
    results = evaluate_model(y_test_inv, y_pred_inv, False)

    return results, y_pred_inv

def fit_lstm(X_train, train, valid, path, epochs=150, dropout=0.2):

    set_tf_seed()

    model = Sequential([
        LSTM(100, activation='relu', input_shape=X_train.shape[-2:], return_sequences=True),
        Flatten(),
        Dense(150, activation='relu'),
        Dropout(dropout),
        Dense(1)
    ])
    best_model = ModelCheckpoint(f"{path}lstm.keras", save_best_only=True, monitor="val_loss", verbose=1)

    metric = [RootMeanSquaredError()]
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=6e-4, amsgrad=True), metrics=metric)

    early_stopping = EarlyStopping(patience=15)
    lr_schedule = LearningRateScheduler(lambda epoch: min(6e-4 * 10**(epoch / 10), 0.005) * (0.96 ** epoch))

    history = model.fit(x = train, validation_data = valid, epochs=epochs,
                                 callbacks=[early_stopping, lr_schedule, best_model])

    return history

def fit_stacked_lstm(X_train, train, valid, path, epochs=150, dropout=0.2):
    
    set_tf_seed()

    model = Sequential([
        LSTM(200, activation='relu', input_shape=X_train.shape[-2:], return_sequences=True),
        LSTM(100, activation='relu', return_sequences=True),
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(dropout),
        Dense(1)
    ])
    best_model = ModelCheckpoint(f"{path}stacked_lstm.keras", save_best_only=True, monitor="val_loss", verbose=1)

    metric = [RootMeanSquaredError()]
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=1e-3, amsgrad=True), metrics=metric)

    early_stopping = EarlyStopping(patience=15)    
    lr_schedule = LearningRateScheduler(lambda epoch: min(1e-3 * 10**(epoch / 10), 0.006) * (0.99 ** epoch))
    history = model.fit(x = train, validation_data = valid, epochs=epochs,
                                 callbacks=[early_stopping, lr_schedule, best_model])

    return history

def fit_cnn_lstm(X_train, train, valid, path, epochs=150, dropout=0.2):

    set_tf_seed()

    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', 
               strides=1, padding='causal', input_shape=X_train.shape[-2:]),
        LSTM(150, activation='relu', return_sequences=True),
        LSTM(100, activation='relu', return_sequences=True),
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(dropout),
        Dense(1)
    ])
    best_model = ModelCheckpoint(f"{path}cnn_lstm.keras", save_best_only=True, monitor="val_loss", verbose=1)

    metric = [RootMeanSquaredError()]
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=1e-3, amsgrad=True), metrics=metric)

    early_stopping = EarlyStopping(patience=15)
    lr_schedule = LearningRateScheduler(lambda epoch: min(1e-3 * 10**(epoch / 10), 0.006) * (0.99 ** epoch))
    history = model.fit(x = train, validation_data = valid, epochs=epochs,
                                 callbacks=[early_stopping, lr_schedule, best_model])

    return history

