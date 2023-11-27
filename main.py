#David, Adam and Efe - June 2023

import numpy as np
import pandas as pd
import time

import preprocessing
import randomforest
import ranking
import lambda_val
import explore
import content_based_filtering
import subset_ndcg

'''
Our main framework consists of a couple of components. First, we load in the data set in main.py
Next, we analyse the raw data set by performing some basic statistical inference and plotting interesting features. 
Furthermore, we clean the raw data set using techniques learned in this course. Also some feature engineering is 
performed in order to get a better end result. These pre-processing/pre-modeling techniques can be found in preprocessing.py
We then start testing different models on the clean data set. As baseline model we use collaborative filtering.
Other models we use are: Randomforest and LamdaMart.
'''

start_time = time.time()

def load_data(preproccesed_data: bool, save: bool = False) -> pd.DataFrame:
        """If load_data function is already run once, for future use can use boolean value == True
        This will load in the cleaned and feature engineered data set instead. """
        
        if preproccesed_data == True:
            print('Clean data is being loaded in..')
            print('-------- - ---------')
            train_clean = pd.read_csv('train_clean_proxies.csv') #pd.read_csv('train_clean_proxies.csv')
            test_clean = pd.read_csv('test_clean_proxies.csv')   #pd.read_csv('test_clean_proxies.csv')
            print('Data loading is finished.')
            print('-------- - ---------')
            
            # This will the loop when using the raw unprocessed data set 
        else:    
            # Load data
            print('Raw data is being loaded in..')
            print('-------- - ---------')
            df_hotel_train_raw = pd.read_csv('training_set_VU_DM.csv')
            df_hotel_test_raw = pd.read_csv('test_set_VU_DM.csv')
            
            # preprocessing data
                # returns a clean data set, both for train and test
            preprocessing.missing_values_overview(df_hotel_train_raw)
            
            print('executing cleaning process alogorithm...')
            print('-------- - ---------')
            # test == boolean value
                #  test = 0, when giving the cleaning algorithm the train set
                #  test = 1, when giving the cleaning algorithmn the test set
            train_clean = preprocessing.cleaning_algo(df_hotel_train_raw, test = 0)
            test_clean = preprocessing.cleaning_algo(df_hotel_test_raw, test = 1)

            #Building proxy for position of a given pid 
            train_clean.loc[:, 'position_proxy'] = randomforest.position_proxy(train_clean, train_clean)
            test_clean.loc[:, 'position_proxy'] = randomforest.position_proxy(train_clean, test_clean)

            #Building proxy/estimation for click bool
            train_clean.loc[:, 'click_proxy'] = randomforest.click_proxy(train_clean, train_clean)
            test_clean.loc[:, 'click_proxy'] = randomforest.click_proxy(train_clean, test_clean)

            print('preprocessing done')
            print('-------- - ---------')

            if save == True:
                # save clean dataframe for future use
                print('Saving csv files...')
                train_clean.to_csv('train_clean.csv', index=False)
                test_clean.to_csv('test_clean.csv', index=False)
                print('Data loading process is finished')

        return train_clean, test_clean



def main():
    """Main function to execute the program.

    Args:
        preproccesed_data (bool): Flag indicating whether the data is preprocessed or not:
                    preproccesed_data = True when predicting actual test set
                    preproccesed_data = False for hyperparameter trainig
        execute (bool): Flag to indicate whether to include subsetted testing and ndcg scoring for the random forest model(s)
        train_clean (pd.DataFrame): cleaned train data using the load_data() function 
        test_clean (pd.DataFrame): cleaned test data using the load_data() function 
        rand_forest_trained (RandomForestClassifier object): trained random forest model with the train_model() function
        test (bool): test = 0 means used on training data, test = 1 means on test data
        training (bool): Flag indicating optimization or pre-specficified parameters:
                    if training = True -> hyper parameter optimization
                    if training = False -> use specified parameters (in lambdaMart) to predict test set and save as csv

    Returns:
        print statements indicating processses
        plots
        dataframes with tables and results
        csv file with kaggle submission file 
    """

    #load and clean data...
    train_clean, test_clean = load_data(preproccesed_data=False)

    #############################

    #data exploration
    print('Data exploration phase:')
    print('-------- - ---------')
    #requires raw dataset for some of the data exploration
    raw_data = pd.read_csv('training_set_VU_DM.csv')
    #descriptives table:
    variables_descr_table = raw_data.select_dtypes(include='number').dropna(axis=1).drop(['srch_id', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id', 'promotion_flag', 'srch_destination_id', 'random_bool', 'click_bool', 'booking_bool'], axis=1)
    print(explore.Descriptives(variables_descr_table).T.to_latex())
    #plots:
    explore.feat_eng_plot(train_clean)
    explore.promotion_effect(raw_data)
    explore.srch_page_position(raw_data)
    explore.countrybook(train_clean)
    explore.hotel_price_distr(train_clean)

    #############################

    #implementing CONTENT-BASED filtering recommender system

    print('Implementing content-based filtering on a subset of the data')
    print('-------- - ---------')
    ranked_df_content_based = content_based_filtering.output(train_clean, test_clean)
    ranked_df_content_based.to_csv('ranked_df_content_based.csv', index=False)

    print('Content-based recommeder system ndcg score')
    print('-------- - ---------')
    ndcg_df_content_based = content_based_filtering.output_in_sample(train_clean)
    ndcg_score_content_based = subset_ndcg.ndcg(actual = ndcg_df_content_based['actual_rating'], scores = ndcg_df_content_based["similarity_score"], k=None)

    print('Content-based recommeder system done.')
    print('-------- - ---------')

    #############################

    #implementing RANDOM FOREST classifier model(s):
    print('Implementing random forest')
    print('-------- - ---------')

    #trying out the algorithm concerned with the random forest(s) on a smaller subset and calculate corresponding ndcg score
    subset_ndcg.subsetting_testing_rf(train_clean, execute=True) #set equal to zero to skip the subset/testing/manual ndcg calculations

    print('Training random forest...')
    rand_forest_trained = randomforest.train_model(train_clean)
    print('Random forest training done.')
    print('-------- - ---------')

    #implementing ranking of test set for kaggle submission
    print('Ranking test set for random forest...')
    ranked_df_submission_random_forest = ranking.rank_full_df(test_clean, rand_forest_trained, 1)
    ranked_df_submission_random_forest.to_csv('ranked_test_random_forest.csv', index=False)
    print('RF ranking done.')
    print('-------- - ---------')
    
    #############################

    #LAMBDAMART model:
        # if training = True -> hyper parameter optimization
        # if training = False -> use specified parameters (in lambdaMart) to predict test set and save as csv
    lambda_val.lambdaMart(train_clean, test_clean, training = False)

    print("Process finished --- %s seconds ---" % (time.time() - start_time))

    #############################

if __name__ == '__main__':
    main()

