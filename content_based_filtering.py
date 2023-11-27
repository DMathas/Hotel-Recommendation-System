import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import preprocessing
# from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split



# subset the data based on srch_id

def subset_srch_id(df, subset_start, subset_end):
    subset = df["srch_id"].unique()[subset_start:subset_end]
    df_subset = df.loc[df["srch_id"].isin(subset)]

    return df_subset


# get similarity matrices

def get_similarity_matrix(df, list, group_by):
    unique_df = df[list].groupby(group_by).mean().reset_index()

    features_matrix = unique_df.drop(group_by, axis=1).values

    sim_matrix = cosine_similarity(features_matrix, features_matrix)

    df_sim_matrix = pd.DataFrame(sim_matrix, index = unique_df[group_by], columns = unique_df[group_by])

    return df_sim_matrix


# get similarity score

def get_similarity_score(row, prop_similarity_matrix):
    prop_id = row['prop_id']
    prop_id_to_compare = row['prop_id_to_compare']
    
    # Handle missing values
    if pd.isnull(prop_id_to_compare):
        return np.nan
    
    try:
        # Get the similarity score from sim_matrix
        similarity_score = prop_similarity_matrix.loc[prop_id, prop_id_to_compare]
    except KeyError:
        similarity_score = np.nan
    
    return similarity_score



def output(df_train, df_test):
    # load the train and test data
    df_hotel_train_clean = preprocessing.cleaning_algo(df_train, 0)
    df_hotel_test_clean = preprocessing.cleaning_algo(df_test, 1)

    # subsetted clean data
    subset_train_clean = subset_srch_id(df_hotel_train_clean, 0, 500)
    subset_test_clean = subset_srch_id(df_hotel_test_clean, 0, 500)

    # combine subsetted data for similarity matrices
    subset_df_clean = pd.concat([subset_train_clean, subset_test_clean], axis=0)
    subset_df_clean = subset_df_clean.reset_index(drop=True)
    subset_df_clean = subset_df_clean.sort_values("srch_id")

    # required lists for similarity matrices
    prop_col_list = ['prop_id','mean_prop_1_sid', 'mean_prop_2_sid', 'mean_prop_review_sid',
                 'mean_prop_star_sid', 'mean_price_usd_pid', 'mean_prop_1_pid',
                 'mean_prop_2_pid', 'mean_prop_review_pid', 'mean_prop_star_pid',
                 'mean_price_usd_pcid', 'mean_prop_1_pcid', 'mean_prop_2_pcid',
                 'mean_prop_review_pcid', 'mean_prop_star_pcid']

    srch_col_list = ['srch_id', 'srch_children_count', 'prop_country_id',
                 'srch_room_count', 'srch_saturday_night_bool','srch_length_of_stay', 
                 'srch_booking_window','mean_prop_1_sid', 
                 'mean_prop_2_sid', 'mean_prop_review_sid','mean_prop_star_sid']
    
    # prop and srch id similarity matrices
    prop_similarity_matrix = get_similarity_matrix(subset_df_clean, list = prop_col_list, group_by = "prop_id")
    srch_similarity_matrix = get_similarity_matrix(subset_df_clean, list = srch_col_list, group_by = "srch_id")


    # extracted datasets for output df
    df_train_extract = subset_df_clean[['srch_id','prop_id','actual_rating']][subset_df_clean[['srch_id','prop_id','actual_rating']]['actual_rating'] != 0].dropna(subset=['actual_rating'])
    df_train_extract = df_train_extract.loc[df_train_extract.groupby("srch_id")["actual_rating"].idxmax()]

    df_test_extract = subset_df_clean.loc[subset_df_clean['actual_rating'].isna(), ['srch_id','prop_id']]

    # prop_id_to_compare for the srch_id that are in both test and train datasets
    prop_id_mapping = df_train_extract.set_index('srch_id')['prop_id'].to_dict()
    df_test_extract["prop_id_to_compare"] = df_test_extract['srch_id'].map(prop_id_mapping)

    # fill the prop_id_to_compare for the srch_ids that are in test but not in train by using the srch_id similarities
    srch_ids_list = df_train_extract['srch_id'].unique()

    missing_values = df_test_extract['prop_id_to_compare'].isnull()

    for srch_id in df_test_extract.loc[missing_values, 'srch_id'].unique():
        similarity_scores = srch_similarity_matrix.loc[srch_id]

        most_similar_srch_id = similarity_scores.loc[similarity_scores.index.isin(srch_ids_list)].idxmax()

        most_similar_prop_id = df_train_extract.loc[(df_train_extract['srch_id'] == most_similar_srch_id),'prop_id'].iloc[0]

        df_test_extract.loc[missing_values & (df_test_extract['srch_id'] == srch_id), 'prop_id_to_compare'] = most_similar_prop_id


    # create a new column with similarity scores
    df_test_extract['similarity_score'] = df_test_extract.apply(lambda row: get_similarity_score(row,prop_similarity_matrix), axis=1) 
    
    # ranking
    sorted_df = df_test_extract.sort_values(by=['srch_id', 'similarity_score'], ascending=[True, False]).reset_index(drop=True)
    output_df = sorted_df[['srch_id', 'prop_id', 'actual_rating', 'similarity_score']]

    return output_df



def output_in_sample(df_train):
    # load the train and test data
    df_hotel_train_clean = preprocessing.cleaning_algo(df_train, 0)

    # subsetted clean data
    subset_train_clean = subset_srch_id(df_hotel_train_clean, 0, 500)

    # train test split
    shuffled_df = subset_train_clean.sample(frac=1, random_state=42)

    train_ratio = 0.6

    train_data, test_data = train_test_split(shuffled_df, train_size=train_ratio, test_size=1-train_ratio)

    # required lists for similarity matrices
    prop_col_list = ['prop_id','mean_prop_1_sid', 'mean_prop_2_sid', 'mean_prop_review_sid',
                 'mean_prop_star_sid', 'mean_price_usd_pid', 'mean_prop_1_pid',
                 'mean_prop_2_pid', 'mean_prop_review_pid', 'mean_prop_star_pid',
                 'mean_price_usd_pcid', 'mean_prop_1_pcid', 'mean_prop_2_pcid',
                 'mean_prop_review_pcid', 'mean_prop_star_pcid']

    srch_col_list = ['srch_id', 'srch_children_count', 'prop_country_id',
                 'srch_room_count', 'srch_saturday_night_bool','srch_length_of_stay', 
                 'srch_booking_window','mean_prop_1_sid', 
                 'mean_prop_2_sid', 'mean_prop_review_sid','mean_prop_star_sid']
    
    # prop and srch id similarity matrices
    prop_similarity_matrix = get_similarity_matrix(subset_train_clean, list = prop_col_list, group_by = "prop_id")
    srch_similarity_matrix = get_similarity_matrix(subset_train_clean, list = srch_col_list, group_by = "srch_id")


    # extracted datasets for output df
    df_train_extract = train_data[['srch_id','prop_id','actual_rating']][train_data[['srch_id','prop_id','actual_rating']]['actual_rating'] != 0].dropna(subset=['actual_rating'])
    df_train_extract = df_train_extract.loc[df_train_extract.groupby("srch_id")["actual_rating"].idxmax()]

    df_test_extract = test_data[['srch_id','prop_id','actual_rating']]

    # prop_id_to_compare for the srch_id that are in both test and train datasets
    prop_id_mapping = df_train_extract.set_index('srch_id')['prop_id'].to_dict()
    df_test_extract["prop_id_to_compare"] = df_test_extract['srch_id'].map(prop_id_mapping)

    # fill the prop_id_to_compare for the srch_ids that are in test but not in train by using the srch_id similarities
    srch_ids_list = df_train_extract['srch_id'].unique()

    missing_values = df_test_extract['prop_id_to_compare'].isnull()

    for srch_id in df_test_extract.loc[missing_values, 'srch_id'].unique():
        similarity_scores = srch_similarity_matrix.loc[srch_id]

        most_similar_srch_id = similarity_scores.loc[similarity_scores.index.isin(srch_ids_list)].idxmax()

        most_similar_prop_id = df_train_extract.loc[(df_train_extract['srch_id'] == most_similar_srch_id),'prop_id'].iloc[0]

        df_test_extract.loc[missing_values & (df_test_extract['srch_id'] == srch_id), 'prop_id_to_compare'] = most_similar_prop_id


    # create a new column with similarity scores
    df_test_extract['similarity_score'] = df_test_extract.apply(lambda row: get_similarity_score(row,prop_similarity_matrix), axis=1)

    # ranking
    sorted_df = df_test_extract.sort_values(by=['srch_id', 'similarity_score'], ascending=[True, False]).reset_index(drop=True)
    output_df = sorted_df[['srch_id', 'prop_id', 'actual_rating', 'similarity_score']]

    return output_df