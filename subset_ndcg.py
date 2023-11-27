import numpy as np
import pandas as pd
from itertools import groupby

import randomforest
import ranking

#NDCG function
def ndcg(actual: pd.DataFrame, scores: pd.DataFrame, k=None) -> float:
    """Calculate the NDCG (Normalized Discounted Cumulative Gain) for a given column (df) 
    containing the actual rating (booked, clicked, or none) and one containing the estimated ranking (booking probabilities)."""

    # Group the data by search ID
    grouped_data = groupby(zip(scores, actual), key=lambda x: x[0])
    
    ndcg_scores = []
    for _, group in grouped_data:
        group = np.array(list(group))
        group_scores = group[:, 0]
        group_relevance = group[:, 1]
        
        # Sort the scores and relevance in descending order
        sorted_indices = np.argsort(group_scores)[::-1]
        sorted_scores = group_scores[sorted_indices]
        sorted_relevance = group_relevance[sorted_indices]
        
        # Calculate DCG (Discounted Cumulative Gain) and IDCG (Ideal Discounted Cumulative Gain) for the group
        dcg = (2 ** sorted_relevance - 1) / np.log2(np.arange(1, len(sorted_scores) + 1) + 1)
        #idcg = (2 ** sorted_relevance[:5] - 1) / np.log2(np.arange(2, 7))
        idcg = (2 ** sorted_relevance[:5] - 1) / np.log2(np.arange(2, 2 + len(sorted_relevance[:5])))

        
        # Calculate NDCG (Normalized Discounted Cumulative Gain) for the group
        ndcg = dcg.sum()
        if idcg.sum() != 0:
            ndcg /= idcg.sum()
        else:
            ndcg = 0
        
        ndcg_scores.append(ndcg)
    
    # Average the NDCG scores over all search IDs
    avg_ndcg = np.mean(ndcg_scores)
    
    # Truncate the result to the specified top-k items if k is given
    if k is not None:
        avg_ndcg = avg_ndcg[:5]
    
    return avg_ndcg



def subsetting_testing_rf(train_data_clean: pd.DataFrame, execute: bool):
    """Quick function that calculates the ndcg scores for subsets of the data in order to provide comparative results. 
       Set execute equal to zero if you want to skip this step."""

    if execute == True:
        print('Iteratively testing the procedure and specifications in terms of ndcg score...')
        print('-------- - ---------')

        #take a subset of df_hotel clean and split 
        full_length = 80000 
        train_prop = 0.5
        train_length = int(full_length * train_prop)

        validation_set_train = train_data_clean.iloc[:train_length, :]
        validation_set_test = train_data_clean.iloc[train_length:full_length, :]

        # validation_set_test.loc[:, 'position_proxy'] = randomforest.position_proxy(validation_set_train, validation_set_test)
        # validation_set_train.loc[:, 'position_proxy'] = randomforest.position_proxy(validation_set_train, validation_set_train)

        # validation_set_test.loc[:, 'click_proxy'] = randomforest.click_proxy(validation_set_train, validation_set_test)
        # validation_set_train.loc[:, 'click_proxy'] = randomforest.click_proxy(validation_set_train, validation_set_train)

        ##### LOOP THAT CHECKS DIFFERENT FEATURE SPECIFICATIONS IN TRAINING AND PREDICTING STAGE ####
        spec_1 = ['prop_starrating', 'prop_location_score2', 'price_usd', 
                                        'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count',
                                        'srch_saturday_night_bool',
                                            'prop_star_review_diff', 'price_diff', 'same_country_srch',
                                            'mean_price_usd_sid', 'mean_prop_1_sid', 'mean_prop_2_sid',
                                            'mean_prop_review_sid', 'mean_prop_star_sid', 'mean_price_usd_pid',
                                            'mean_prop_1_pid', 'mean_prop_2_pid', 'mean_prop_review_pid',
                                            'mean_prop_star_pid', 'mean_price_usd_pcid', 'mean_prop_1_pcid',
                                            'mean_prop_2_pcid', 'mean_prop_review_pcid', 'mean_prop_star_pcid']


        spec_2 = ['prop_starrating', 'prop_location_score2', 'price_usd', 
                                        'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count',
                                        'srch_saturday_night_bool',
                                            'prop_star_review_diff', 'price_diff', 'same_country_srch',
                                            'mean_price_usd_sid', 'mean_prop_1_sid', 'mean_prop_2_sid',
                                            'mean_prop_review_sid', 'mean_prop_star_sid', 'mean_price_usd_pid',
                                            'mean_prop_1_pid', 'mean_prop_2_pid', 'mean_prop_review_pid',
                                            'mean_prop_star_pid', 'mean_price_usd_pcid', 'mean_prop_1_pcid',
                                            'mean_prop_2_pcid', 'mean_prop_review_pcid', 'mean_prop_star_pcid', 'position_proxy']

        spec_3 = ['prop_starrating', 'prop_location_score1', 'prop_location_score2', 'price_usd', 
                                        'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count',
                                        'srch_saturday_night_bool',
                                            'prop_star_review_diff', 'price_diff', 'same_country_srch',
                                            'mean_price_usd_sid', 'mean_prop_1_sid', 'mean_prop_2_sid',
                                            'mean_prop_review_sid', 'mean_prop_star_sid', 'mean_price_usd_pid',
                                            'mean_prop_1_pid', 'mean_prop_2_pid', 'mean_prop_review_pid',
                                            'mean_prop_star_pid', 'mean_price_usd_pcid', 'mean_prop_1_pcid',
                                            'mean_prop_2_pcid', 'mean_prop_review_pcid', 'mean_prop_star_pcid', 'position_proxy']

        spec_4 = ['prop_starrating', 'prop_location_score2', 'price_usd', 
                                        'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count',
                                        'srch_saturday_night_bool',
                                            'prop_star_review_diff', 'price_diff', 'same_country_srch',
                                            'mean_price_usd_sid', 'mean_prop_1_sid', 'mean_prop_2_sid',
                                            'mean_prop_review_sid', 'mean_prop_star_sid', 'mean_price_usd_pid',
                                            'mean_prop_1_pid', 'mean_prop_2_pid', 'mean_prop_review_pid',
                                            'mean_prop_star_pid', 'mean_price_usd_pcid', 'mean_prop_1_pcid',
                                            'mean_prop_2_pcid', 'mean_prop_review_pcid', 'mean_prop_star_pcid', 'position_proxy', 'click_proxy']

        spec_5 = ['prop_starrating', 'prop_location_score2', 'price_usd', 
                                        'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count',
                                        'srch_saturday_night_bool',
                                            'prop_star_review_diff', 'price_diff', 'same_country_srch',
                                            'mean_price_usd_sid', 'mean_prop_1_sid', 'mean_prop_2_sid',
                                            'mean_prop_review_sid', 'mean_prop_star_sid', 'mean_price_usd_pid',
                                            'mean_prop_1_pid', 'mean_prop_2_pid', 'mean_prop_review_pid',
                                            'mean_prop_star_pid', 'mean_price_usd_pcid', 'mean_prop_1_pcid',
                                            'mean_prop_2_pcid', 'mean_prop_review_pcid', 'mean_prop_star_pcid', 'position_proxy', 'click_proxy', 
                                            'orig_destination_distance', 'comp_sum2_8', 'comp_sum5']


        specifications_subset = [spec_1, spec_2, spec_3, spec_4, spec_5]

        #train the subset and predict/rank:
        for i in specifications_subset:
            val_set_model = randomforest.train_model(validation_set_train, i)
            val_set_ranked = ranking.rank_full_df(validation_set_test, val_set_model, 0, i)

            print(f'{i} produces a NDCG score of:')
            print(ndcg(val_set_ranked['actual'], val_set_ranked['book_prob']))
            print('-------- - ---------')


    



