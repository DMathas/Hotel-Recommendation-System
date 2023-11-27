import pandas as pd

#function that ranks property id's given one search id
def prop_rank(search_id: float, data: pd.DataFrame, trained_model, test: bool, features: list) -> pd.DataFrame:
    
    """Rank the properties belonging to a user search on the likeliness that the property will be booked. Here,
       you should start with listing the property most likely to be booked. 
       test = 0 means used on training data, test = 1 means on test data.. """

    sid_subset_mask = data['srch_id'] == search_id

    X_predict = data[sid_subset_mask][features]

    if test == 0:
      return pd.DataFrame({'search_id': data[sid_subset_mask]['srch_id'], 
                                'prop_id': data[sid_subset_mask]['prop_id'],
                                'book_prob': trained_model.predict_proba(X_predict)[:, 1],
                                'actual': data[sid_subset_mask]['actual_rating']}).sort_values(by='book_prob', ascending=False)
    if test == 1:
      return pd.DataFrame({'search_id': data[sid_subset_mask]['srch_id'], 
                                'prop_id': data[sid_subset_mask]['prop_id'],
                                'book_prob': trained_model.predict_proba(X_predict)[:, 1]}).sort_values(by='book_prob', ascending=False)
    
#apply to entire dataframe:
def rank_full_df(data, trained_model, test: bool, features: list = ['prop_starrating', 'prop_location_score2', 'price_usd', 
                                 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count',
                                 'srch_saturday_night_bool',
                                    'prop_star_review_diff', 'price_diff', 'same_country_srch',
                                    'mean_price_usd_sid', 'mean_prop_1_sid', 'mean_prop_2_sid',
                                    'mean_prop_review_sid', 'mean_prop_star_sid', 'mean_price_usd_pid',
                                    'mean_prop_1_pid', 'mean_prop_2_pid', 'mean_prop_review_pid',
                                    'mean_prop_star_pid', 'mean_price_usd_pcid', 'mean_prop_1_pcid',
                                    'mean_prop_2_pcid', 'mean_prop_review_pcid', 'mean_prop_star_pcid', 'position_proxy', 'click_proxy', 
                                    'orig_destination_distance', 'comp_sum2_8', 'comp_sum5']) -> pd.DataFrame:
    
    """Creates the entire ranked dataframe. test = 0 means used on training data, test = 1 means on test data."""

    grouped = data.groupby('srch_id')
    ranked_dfs = grouped.apply(lambda x: prop_rank(x.name, x, trained_model, test, features))

    if test == 0:
      ranked_dfs = ranked_dfs[['search_id', 'prop_id', 'book_prob', 'actual']]
      ranked_dfs.columns = ['srch_id', 'prop_id', 'book_prob', 'actual']

    if test == 1:
      ranked_dfs = ranked_dfs[['search_id', 'prop_id']]
      ranked_dfs.columns = ['srch_id', 'prop_id']  

    return ranked_dfs.reset_index(drop=True)