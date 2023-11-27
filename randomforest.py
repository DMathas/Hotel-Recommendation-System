import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

#building a proxy for position variable:
def position_proxy(train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
    """Predicts a proxy for the position of a pid on the site."""
    train_data_subset = train_data.iloc[:100000, :]
    X_train_proxy = train_data_subset[['prop_starrating', 'prop_location_score2', 'price_usd', 
                                 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count',
                                 'srch_saturday_night_bool',
                                    'prop_star_review_diff', 'price_diff', 'same_country_srch',
                                    'mean_price_usd_sid', 'mean_prop_1_sid', 'mean_prop_2_sid',
                                    'mean_prop_review_sid', 'mean_prop_star_sid', 'mean_price_usd_pid',
                                    'mean_prop_1_pid', 'mean_prop_2_pid', 'mean_prop_review_pid',
                                    'mean_prop_star_pid', 'mean_price_usd_pcid', 'mean_prop_1_pcid',
                                    'mean_prop_2_pcid', 'mean_prop_review_pcid', 'mean_prop_star_pcid',
                                    'orig_destination_distance', 'comp_sum2_8', 'comp_sum5']]
    y_train_proxy = train_data_subset['position']
    
    
    #Mposition_proxy = LinearRegression()
    Mposition_proxy = RandomForestRegressor(n_jobs=-1)
    Mposition_proxy.fit(X_train_proxy, y_train_proxy)


    X_test_proxy = test_data[['prop_starrating', 'prop_location_score2', 'price_usd', 
                                 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count',
                                 'srch_saturday_night_bool',
                                    'prop_star_review_diff', 'price_diff', 'same_country_srch',
                                    'mean_price_usd_sid', 'mean_prop_1_sid', 'mean_prop_2_sid',
                                    'mean_prop_review_sid', 'mean_prop_star_sid', 'mean_price_usd_pid',
                                    'mean_prop_1_pid', 'mean_prop_2_pid', 'mean_prop_review_pid',
                                    'mean_prop_star_pid', 'mean_price_usd_pcid', 'mean_prop_1_pcid',
                                    'mean_prop_2_pcid', 'mean_prop_review_pcid', 'mean_prop_star_pcid', 
                                    'orig_destination_distance', 'comp_sum2_8', 'comp_sum5']] #ADD COLUMNS!
    

    return Mposition_proxy.predict(X_test_proxy)

#building a proxy for position variable:
def click_proxy(train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
    """Predicts a proxy for the position of a pid on the site."""
    train_data_subset = train_data.iloc[:250000, :]

    X_train_proxy = train_data_subset[['prop_starrating', 'prop_location_score2', 'price_usd', 
                                 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count',
                                 'srch_saturday_night_bool',
                                    'prop_star_review_diff', 'price_diff', 'same_country_srch',
                                    'mean_price_usd_sid', 'mean_prop_1_sid', 'mean_prop_2_sid',
                                    'mean_prop_review_sid', 'mean_prop_star_sid', 'mean_price_usd_pid',
                                    'mean_prop_1_pid', 'mean_prop_2_pid', 'mean_prop_review_pid',
                                    'mean_prop_star_pid', 'mean_price_usd_pcid', 'mean_prop_1_pcid',
                                    'mean_prop_2_pcid', 'mean_prop_review_pcid', 'mean_prop_star_pcid', 
                                    'orig_destination_distance', 'comp_sum2_8', 'comp_sum5']]
    y_train_proxy = train_data_subset['click_bool']
    
    
    #Mposition_proxy = LinearRegression()
    Mposition_proxy = RandomForestRegressor(n_jobs=-1)
    Mposition_proxy.fit(X_train_proxy, y_train_proxy)

    X_test_proxy = test_data[['prop_starrating', 'prop_location_score2', 'price_usd', 
                                 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count',
                                 'srch_saturday_night_bool',
                                    'prop_star_review_diff', 'price_diff', 'same_country_srch',
                                    'mean_price_usd_sid', 'mean_prop_1_sid', 'mean_prop_2_sid',
                                    'mean_prop_review_sid', 'mean_prop_star_sid', 'mean_price_usd_pid',
                                    'mean_prop_1_pid', 'mean_prop_2_pid', 'mean_prop_review_pid',
                                    'mean_prop_star_pid', 'mean_price_usd_pcid', 'mean_prop_1_pcid',
                                    'mean_prop_2_pcid', 'mean_prop_review_pcid', 'mean_prop_star_pcid', 
                                    'orig_destination_distance', 'comp_sum2_8', 'comp_sum5']]
    
    # X_test_proxy = X_test_proxy.astype(float)
    # X_test_proxy = scaler.fit_transform(X_test_proxy)


    return Mposition_proxy.predict(X_test_proxy)


def train_model(data: pd.DataFrame, features: list = ['prop_starrating', 'prop_location_score2', 'price_usd', 
                                 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count',
                                 'srch_saturday_night_bool',
                                    'prop_star_review_diff', 'price_diff', 'same_country_srch',
                                    'mean_price_usd_sid', 'mean_prop_1_sid', 'mean_prop_2_sid',
                                    'mean_prop_review_sid', 'mean_prop_star_sid', 'mean_price_usd_pid',
                                    'mean_prop_1_pid', 'mean_prop_2_pid', 'mean_prop_review_pid',
                                    'mean_prop_star_pid', 'mean_price_usd_pcid', 'mean_prop_1_pcid',
                                    'mean_prop_2_pcid', 'mean_prop_review_pcid', 'mean_prop_star_pcid', 'position_proxy', 'click_proxy', 
                                    'orig_destination_distance', 'comp_sum2_8', 'comp_sum5']
                                    ):
   
    
    """COnstructs and trains the random forest model."""
    rf_model_TRAIN = RandomForestClassifier(n_jobs=-1)
    X = data[features]

    y = data['booking_bool']
    rf_model_TRAIN.fit(X, y)
    return rf_model_TRAIN