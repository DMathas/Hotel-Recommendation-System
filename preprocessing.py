import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

def missing_values_overview(dataframe):
    #General missing values overview:
    percent_missing = dataframe.isnull().mean()
    total_missing = dataframe.isnull().sum()
    missing_value_df = pd.DataFrame({
                                    'percent_missing': percent_missing,
                                    'missing_count': total_missing})
    print(missing_value_df)

    #Corresponding figure:
    plt.figure(figsize=(11, 2))
    plt.bar(range(len(percent_missing)), percent_missing)
    plt.grid(linestyle='--', linewidth=.2)
    plt.xticks(range(len(percent_missing))[2:], dataframe.columns[2:], fontsize=3, rotation=45)
    plt.ylabel('Missing value proportion in %', fontsize=5)
    plt.subplots_adjust(bottom=0.15)
    plt.axhline(0.66, color='red', linestyle='--', linewidth=0.3, label='Proposed inclusion cutoff')
    plt.legend(loc='upper right', ncol=1, frameon=False, markerscale=0, fontsize=5)
    plt.show()


#Cleaning step:
def cleaning_algo(data: pd.DataFrame, test: bool) -> pd.DataFrame:
    """Performs the essential cleaning steps in order to 
        prepare the data for training/predicting the ranking system. 
        test = 0 means used on training data, test = 1 means on test data."""

    #delete columns with more than 25% missing values
    percent_missing = data.isnull().mean()
    #cutoff_mask = percent_missing < 0.25
    cutoff_mask = percent_missing < 0.66

    data = data.loc[:, cutoff_mask]

    #filter the outliers from each (numeric) column
    numeric_columns = data.select_dtypes(include=np.number).columns
    if test == 0:
        numeric_columns = list(set(numeric_columns) - set(['prop_id', 'srch_id', 'booking_bool']))
    if test == 1:
        numeric_columns = list(set(numeric_columns) - set(['prop_id', 'srch_id']))    
    non_numeric_columns = data.select_dtypes(exclude=np.number).columns
    non_numeric_columns = list(non_numeric_columns)
    if test == 0:
        non_numeric_columns.extend(['prop_id', 'srch_id', 'booking_bool'])
    if test == 1:
        non_numeric_columns.extend(['prop_id', 'srch_id'])    
    numeric_data = data[numeric_columns]
    Q1 = numeric_data.quantile(0.01) #OF TOCH 0.1??
    Q3 = numeric_data.quantile(0.99)
    IQR = Q3 - Q1
    numeric_data = numeric_data[~((numeric_data < (Q1 - 1.5 * IQR)) |(numeric_data > (Q3 + 1.5 * IQR))).any(axis=1)]
    data = data[non_numeric_columns].join(numeric_data)

    #add inv and rate columns for 2 and 5
    data.loc[:, 'comp_sum2_8'] = data['comp2_rate'] + data['comp2_inv'] + data['comp8_rate'] + data['comp8_inv']
    data.loc[:, 'comp_sum5'] = data['comp5_rate'] + data['comp5_inv']
    data = data.drop(['comp2_rate', 'comp2_inv', 'comp5_rate', 'comp5_inv', 'comp8_rate', 'comp8_inv'], axis=1)
    
    #impute the remaining columns that have some (less than 25% missing values) NA's

    data.loc[:, 'prop_location_score2'] = data['prop_location_score2'].fillna(-1)

    if data.isnull().any().any() == True:
        any_na_mask = data.isnull().any()
        if len(data) >= 1000000:
            knn_train_subset = data.sample(frac=0.0005, random_state=42)
            knn_imp = KNNImputer(n_neighbors=3)
            knn_imp.fit(knn_train_subset.loc[:, any_na_mask])
        if len(data) < 1000000:
            knn_train_subset = data.sample(frac=0.005, random_state=42)
            knn_imp = KNNImputer(n_neighbors=3)
            knn_imp.fit(knn_train_subset.loc[:, any_na_mask])
        #immpute the missing values in the entire dataset
        data.loc[:, any_na_mask] = knn_imp.transform(data.loc[:, any_na_mask])

    #feature engineered columns:
    data.loc[:, 'prop_star_review_diff'] = data['prop_starrating'] - data['prop_review_score']
    data.loc[:, 'price_diff'] = data['price_usd'] - np.exp(data['prop_log_historical_price'])
    data.loc[:, 'same_country_srch'] = data['prop_country_id'] == data['visitor_location_country_id']
    #sid based means
    data.loc[:, 'mean_price_usd_sid'] = data.groupby('srch_id')['price_usd'].transform('mean')
    data.loc[:, 'mean_prop_1_sid'] = data.groupby('srch_id')['prop_location_score1'].transform('mean')
    data.loc[:, 'mean_prop_2_sid'] = data.groupby('srch_id')['prop_location_score2'].transform('mean')
    data.loc[:, 'mean_prop_review_sid'] = data.groupby('srch_id')['prop_review_score'].transform('mean')
    data.loc[:, 'mean_prop_star_sid'] = data.groupby('srch_id')['prop_starrating'].transform('mean')
    #pid based means
    data.loc[:, 'mean_price_usd_pid'] = data.groupby('prop_id')['price_usd'].transform('mean')
    data.loc[:, 'mean_prop_1_pid'] = data.groupby('prop_id')['prop_location_score1'].transform('mean')
    data.loc[:, 'mean_prop_2_pid'] = data.groupby('prop_id')['prop_location_score2'].transform('mean')
    data.loc[:, 'mean_prop_review_pid'] = data.groupby('prop_id')['prop_review_score'].transform('mean')
    data.loc[:, 'mean_prop_star_pid'] = data.groupby('prop_id')['prop_starrating'].transform('mean')
    #property cid based means 
    data.loc[:, 'mean_price_usd_pcid'] = data.groupby('prop_country_id')['price_usd'].transform('mean')
    data.loc[:, 'mean_prop_1_pcid'] = data.groupby('prop_country_id')['prop_location_score1'].transform('mean')
    data.loc[:, 'mean_prop_2_pcid'] = data.groupby('prop_country_id')['prop_location_score2'].transform('mean')
    data.loc[:, 'mean_prop_review_pcid'] = data.groupby('prop_country_id')['prop_review_score'].transform('mean')
    data.loc[:, 'mean_prop_star_pcid'] = data.groupby('prop_country_id')['prop_starrating'].transform('mean')

    #add actual rating conditional coliumn:
    if test == 0:
        data['actual_rating'] = np.zeros(len(data))
        data.loc[data['click_bool'] == 1, 'actual_rating'] = 1
        data.loc[data['booking_bool'] == 1, 'actual_rating'] = 5

    return data

