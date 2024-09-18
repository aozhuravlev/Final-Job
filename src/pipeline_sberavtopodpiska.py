import pandas as pd
import dill
import csv
import requests
import datetime
import pygame
import time

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector


def playsound():
    pygame.mixer.init()
    pygame.mixer.music.load("finished_.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


def print_status_and_duration(function_name, start, end):
    total_time = round((end - start))
    seconds = int(total_time % 60)
    minutes = int(total_time // 60)
    print(f"{function_name:<25} -- OK. Time elapsed: {minutes} min {seconds} sec")


def data_loading():
    function_name = 'Data loading and merging'
    start_time = time.time()
    path = 'c:/Users/User/skillbox/data_for_final_job/'
    ga_hits = path + "ga_hits.csv"
    df_hits = pd.read_csv(ga_hits, dtype={"hit_time": "float64", "hit_number": "Int32"})
    df_hits = df_hits[["session_id", "event_action"]]
    ga_sessions = path + "ga_sessions.csv"
    df_sessions = pd.read_csv(ga_sessions, dtype={"visit_number": "Int32"}, low_memory=False)
    df = pd.merge(left=df_sessions, right=df_hits, on='session_id', how='inner')
    end_time = time.time()
    print_status_and_duration(function_name, start_time, end_time)

    columns_2_drop = [
        'device_model',
        'utm_keyword',
        'utm_source',
        'utm_medium',
        'utm_campaign',
        'utm_adcontent',
        'client_id',
    ]

    return df.drop(columns=columns_2_drop)


def binarizing_target(df):
    df = df.copy()
    function_name = 'Binarizing target'
    start_time = time.time()
    action_list = [
        'sub_car_claim_click',
        'sub_car_claim_submit_click',
        'sub_open_dialog_click',
        'sub_custom_question_submit_click',
        'sub_call_number_click',
        'sub_callback_submit_click',
        'sub_submit_success',
        'sub_car_request_submit_click'
    ]

    df.event_action = df.event_action.apply(lambda x: 1 if x in action_list else 0)

    end_time = time.time()
    print_status_and_duration(function_name, start_time, end_time)
    return df


def drop_duplicates(df):
    df = df.copy()
    function_name = 'Duplicates dropped'
    start_time = time.time()
    df = df.sort_values(by=["event_action"], ascending=False)
    df = df.drop_duplicates(subset=["session_id"], keep="first")
    end_time = time.time()
    print_status_and_duration(function_name, start_time, end_time)
    return df    


# def data_preparation(df):
def filling_device_os(df):
    df = df.copy()
    function_name = 'Device OS filling'
    start_time = time.time()

    def fill_device_os(row):
        if pd.isna(row.device_os):
            if row.device_brand == 'Apple':
                return 'iOS'
            elif row.device_category == 'mobile':
                return 'Android'
            elif row.device_category == 'desktop':
                return 'Windows'
            else:
                return 'other'
        else:
            return row['device_os']

    df['device_os'] = df.apply(fill_device_os, axis=1)
    end_time = time.time()
    print_status_and_duration(function_name, start_time, end_time)
    return df


def filling_device_brand(df):
    df = df.copy()
    function_name = 'Device brand filling'
    start_time = time.time()

    def fill_device_brand(row):
        if pd.isna(row.device_brand):
            if row.device_os == 'Macintosh':
                return 'Apple'
            elif row.device_os == "Windows" or row.device_os == "Linux":
                return 'custom'
            else:
                return 'other'
        else:
            return row['device_brand']

    df['device_brand'] = df.apply(fill_device_brand, axis=1)
    end_time = time.time()
    print_status_and_duration(function_name, start_time, end_time)
    return df


# def feature_engineering(df):
def datetime_conversion(df):
    df = df.copy()
    function_name = 'Date & time conversion'
    start_time = time.time()
    df.visit_date = pd.to_datetime(df.visit_date)

    df["visit_month"] = df.visit_date.dt.month
    df["visit_day"] = df.visit_date.dt.day
    df["visit_dayofweek"] = df.visit_date.dt.day_of_week
    df["visit_year"] = df.visit_date.dt.year
    end_time = time.time()
    print_status_and_duration(function_name, start_time, end_time)
    
    return df.drop(columns=['visit_year'])


def day_categorization(df):
    df = df.copy()
    function_name = 'Days categorized'
    start_time = time.time()

    def get_holidays_list():
        holidays_list = []
        with open('holidays_list.csv', mode='r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                holiday_row = tuple(map(int, row))
                holidays_list.append(holiday_row)

        return holidays_list

    holidays_list = get_holidays_list()
    df["day_category"] = df.apply(
        lambda x: 'holiday' if (x.visit_month, x.visit_day) in holidays_list
        else ('workday' if x.visit_dayofweek < 6 else 'weekend'), axis=1
    )
    end_time = time.time()
    print_status_and_duration(function_name, start_time, end_time)
    return df.drop(columns=['visit_dayofweek', 'visit_day'])


def season_categorization(df):
    df = df.copy()
    function_name = 'Season categorized'
    start_time = time.time()
    df["season"] = df.visit_month.apply(
        lambda x: 'winter' if x == 12 or x == 1 or x == 2
        else ('spring' if x < 6 else ('summer' if x < 9 else 'autumn'))
    )
    end_time = time.time()
    print_status_and_duration(function_name, start_time, end_time)
    return df.drop(columns=['visit_month'])


def time_categorization(df):
    df = df.copy()
    function_name = 'Time categorized'
    start_time = time.time()
    df["visit_time_category"] = df.visit_time.apply(
        lambda x: 'night' if int(x[:2]) < 5
        else ('morning' if int(x[:2]) < 8
              else ('morning_rush' if int(x[:2]) < 10
                    else ('workhour' if int(x[:2]) < 17
                          else ('evening_rush' if int(x[:2]) < 19
                                else 'evening'
                                )
                          )
                    )
              )
    )
    end_time = time.time()
    print_status_and_duration(function_name, start_time, end_time)
    return df.drop(columns=['visit_time'])


def screen_categorization(df):
    df = df.copy()
    function_name = 'Screen categorized'
    start_time = time.time()

    def categorize_device(row):
        category = row['device_category']
        area = row['device_screen_area']

        if category == 'mobile':
            if area < 130000:
                return 'small_mobile'
            elif area < 200000:
                return 'medium_mobile'
            else:
                return 'big_mobile'
        elif category == 'desktop':
            if area < 500000:
                return 'small_desktop'
            elif area < 1000000:
                return 'medium_desktop'
            else:
                return 'big_desktop'
        elif category == 'tablet':
            if area < 300000:
                return 'small_tablet'
            elif area < 600000:
                return 'medium_tablet'
            else:
                return 'big_tablet'
        else:
            return 'unknown'

    df['device_screen_area'] = df.device_screen_resolution.apply(
        lambda x: int(x.split("x")[0]) * int(x.split("x")[1])
    )
    df['screen_category'] = df.apply(categorize_device, axis=1)
    end_time = time.time()
    print_status_and_duration(function_name, start_time, end_time)
    return df.drop(columns=['device_screen_resolution'])


def city_type_categorization(df):
    df = df.copy()
    function_name = 'City type categorized'
    start_time = time.time()

    def get_city_type_dict():
        csv_filename = 'updated_city_type.csv'
        city_type = {}

        with open(csv_filename, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                key = row[0]
                value = row[1]
                city_type[key] = value

        return city_type

    def get_city_type_from_geonames(city):
        username = 'azhuravlev'
        url = f"http://api.geonames.org/searchJSON?q={city}&maxRows=1&username={username}"
        response = requests.get(url)
        data = response.json()
        try:
            if data['totalResultsCount'] > 0:
                description = data['geonames'][0].get('fcodeName').split()
                if 'first-order' in description:
                    return 'first'
                elif 'second-order' in description:
                    return 'second'
                elif 'third-order' in description:
                    return 'third'
                elif 'capital' in description:
                    return 'capital'
                elif 'abandoned' in description:
                    return 'abandoned'
                else:
                    return 'place'
            else:
                return 'abandoned place'
        except Exception:
            return 'abandoned place'

    def determine_city_type(city_name):
        if city_name in city_type_dict.keys():
            city_type = city_type_dict[city_name]
        else:
            city_type = get_city_type_from_geonames(city_name)
        return city_type

    city_type_dict = get_city_type_dict()
    df['city_type'] = df.geo_city.apply(lambda x: determine_city_type(x))
    end_time = time.time()
    print_status_and_duration(function_name, start_time, end_time)
    return df.drop(columns=['geo_city'])


def former_USSR_categorization(df):
    df = df.copy()
    function_name = 'Country categorized'
    start_time = time.time()
    former_ussr_countries = ['Armenia', 'Azerbaijan', 'Belarus', 'Estonia', 'Georgia',
                             'Kazakhstan', 'Kyrgyzstan', 'Latvia', 'Lithuania', 'Moldova',
                             'Tajikistan', 'Turkmenistan', 'Ukraine', 'Uzbekistan', 'Russia']
    df['USSR'] = df.geo_country.apply(
        lambda x: 1 if x in former_ussr_countries else 0
    )
    end_time = time.time()
    print_status_and_duration(function_name, start_time, end_time)
    return df.drop(columns=['geo_country'])


def dev_brand_categorization(df):
    df = df.copy()
    function_name = 'Dev brand categorized'
    start_time = time.time()
    dev_brand_count = df.device_brand.value_counts().to_dict()
    df["device_brand_cat"] = df["device_brand"].apply(
        lambda x: x if dev_brand_count[x] > 100000 else 'other'
    )
    end_time = time.time()
    print_status_and_duration(function_name, start_time, end_time)
    return df.drop(columns=['device_brand'])


def dev_os_categorization(df):
    df = df.copy()
    function_name = 'Dev OS categorized'
    start_time = time.time()
    dev_os_count = df.device_os.value_counts().to_dict()
    df['device_os_cat'] = df.device_os.apply(
        lambda x: x if dev_os_count[x] > 1000 else 'other'
    )
    end_time = time.time()
    print_status_and_duration(function_name, start_time, end_time)
    return df.drop(columns=['device_os'])


def dev_browser_categorization(df):
    df = df.copy()
    function_name = 'Dev browser categorized'
    start_time = time.time()
    dev_browser_count = df.device_browser.value_counts().to_dict()
    df["device_browser_cat"] = df["device_browser"].apply(
        lambda x: x if dev_browser_count[x] > 100000 else 'other'
    )
    end_time = time.time()
    print_status_and_duration(function_name, start_time, end_time)
    return df.drop(columns=['device_browser'])


def filter_data(df):
    df = df.copy()
    columns_to_drop = [
        'session_id',
    ]
    print("\nUnnecessary features dropped OK. Proceeding to OHE and Scaler. See you soon...\n")
    return df.drop(columns=columns_to_drop)


def main():
    start_time_all = time.time()
    pipeline_name = 'SberAutoSubscription - User Action Prediction Pipeline'
    equal_signs = '=' * len(pipeline_name)
    print(f"{equal_signs}\n{pipeline_name}\n{equal_signs}")

    initial_df = data_loading()
    
    df_processor = Pipeline(steps=[
        ('duplicates_dropping', FunctionTransformer(drop_duplicates)),
        ('binarizing_target', FunctionTransformer(binarizing_target)),
    ])
    
    df = df_processor.fit_transform(initial_df)

    X = df.drop(columns=["event_action"])
    y = df["event_action"]

    numerical_features = make_column_selector(dtype_include=["int64", "float64", "Int32"])
    categorical_features = make_column_selector(dtype_include=["object"])

    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    data_preparation = Pipeline(steps=[
        ('filling_device_os', FunctionTransformer(filling_device_os)),
        ('filling_device_brand', FunctionTransformer(filling_device_brand)),
    ])

    feature_engineering = Pipeline(steps=[
        ('datetime_conversion', FunctionTransformer(datetime_conversion)),
        ('day_categorization', FunctionTransformer(day_categorization)),
        ('season_categorization', FunctionTransformer(season_categorization)),
        ('time_categorization', FunctionTransformer(time_categorization)),
        ('screen_categorization', FunctionTransformer(screen_categorization)),
        ('city_type_categorization', FunctionTransformer(city_type_categorization)),
        ('former_USSR_categorization', FunctionTransformer(former_USSR_categorization)),
        ('dev_brand_categorization', FunctionTransformer(dev_brand_categorization)),
        ('dev_os_categorization', FunctionTransformer(dev_os_categorization)),
        ('dev_browser_categorization', FunctionTransformer(dev_browser_categorization)),
        ('filter', FunctionTransformer(filter_data)),
    ])

    col_transformer = ColumnTransformer(transformers=[
        ("numerical", numerical_transformer, numerical_features),
        ("categorical", categorical_transformer, categorical_features)
    ])

    preprocessor = Pipeline(steps=[
        ('data_preparation', data_preparation),
        ('feature_engineering', feature_engineering),
        ("col_transformer", col_transformer),
    ])

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        reg_lambda=1,
        gamma=1,
        alpha=0.1,
        tree_method='hist',
        eval_metric='auc',
    )

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])

    step_name = 'The entire task performed'
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    fitted_pipe = pipe.fit(x_train, y_train)
    score = round((round((roc_auc_score(y_test, fitted_pipe.predict_proba(x_test)[:, 1])), 6) * 100), 2)
    
    fully_fitted_pipe = pipe.fit(X, y)
    with open('user_action_prediction_pipeline.pkl', 'wb') as file:
        dill.dump({
            'model': fully_fitted_pipe,
            'metadata': {
                'name': 'SberAutoSubscription - User Action Prediction Pipeline',
                'author': 'Aleksey Zhuravlev',
                'version': 1,
                'date': datetime.datetime.now(),
                'type': type(model).__name__,
                'accuracy': score
            }
        }, file, recurse=True)

    end_time_all = time.time()
    print_status_and_duration(step_name, start_time_all, end_time_all)
    print(f'{equal_signs}\nThe roc_auc score of the model is: {score} % \
          \n{equal_signs}')

    playsound()


if __name__ == '__main__':
    main()
