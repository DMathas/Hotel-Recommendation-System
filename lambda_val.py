import pandas as pd
import numpy as np
from itertools import groupby
import pyltr

def lambdaMart(df_train: pd.DataFrame, df_test: pd.DataFrame, training: bool) -> pd.DataFrame:
    """Trains the lambdaMart model and builds the corresponding dataframe to be submitted."""

    if training == True:
        # x are the features for training
        x = df_train.drop(['date_time', 'actual_rating', 'srch_id', 'booking_bool', 'click_bool', 'position'], axis=1)
        print('x:')
        print(x)
        print('')

        # y is the target variable
        y = df_train['actual_rating']
        # qid is the grouping parameter
        qid = df_train['srch_id']

        # split x and y into train and validation set
            # interested in taking roughly 20% of the train set as validation set
            # to not have an overlap between srch_id's between train and validation take entire srch_id that is on the 80% mark

        x_train = x[:80017]   # last row of srch_id = 53512
        y_train = y[:80017]
        qid_train = qid[:80017]
        print('hier')
        print(x_train.shape)

        x_val = x[80017:1000000]     # first row of next srch_id
        y_val = y[80017:1000000]
        qid_val = qid[80017:1000000]


        # GRID SEARCH #
        # default value is always included in the list of possibilities
            # n_estimators: large number normally leads to better performance (source: https://github.com/jma127/pyltr/blob/master/pyltr/models/lambdamart.py)
        n_estimators = [100, 250, 500, 1000]
        min_samples_leafs = [1, 10, 50, 75]
        max_depth = [3, 10, 50, 100]
        learning_rate = [0.01, 0.1]
        # k = 5 is so that it's consistent with kaggle competition NDCG score
        metric = pyltr.metrics.NDCG(k=5)

        score = []
        best_score = 0

        for n in n_estimators:
            for leafs in min_samples_leafs:
                for depth in max_depth:
                    for learning in learning_rate:
                        print(f'n: {n}')
                        print(f'leafs: {leafs}')
                        print(f'max depth: {depth}')
                        print(f'learning rate: {learning}')

                        model = pyltr.models.LambdaMART(
                            metric=metric,
                            n_estimators = n,
                            min_samples_leaf = leafs,
                            max_depth = depth,
                            learning_rate= learning,
                            verbose = 1,
                            random_state=42)
                    
                        print('fit model')
                        model.fit(x_train, y_train, qid_train)

                        print('now predicting')
                        y_pred = model.predict(x_val)
                        print(y_pred)

                        print('Random ranking:', metric.calc_mean_random(qid_val, y_val))
                        print('Our model:', metric.calc_mean(qid_val, y_val.values, y_pred))

                        score.append(metric.calc_mean(qid_val, y_val.values, y_pred))
                        if metric.calc_mean(qid_val, y_val.values, y_pred) > best_score:
                            best_score = metric.calc_mean(qid_val, y_val.values, y_pred)
                            print(f'new best score with: n = {n_estimators}, leafs: {min_samples_leafs}, depth: {depth} learning rate: {learning}')

        print('best score:')
        print(best_score)
        print('overall scores:')
        print(score)
        print('')
    
    else:
        print('Initializing Kaggle test')
        x_train = df_train.drop(['date_time', 'actual_rating', 'srch_id', 'booking_bool', 'click_bool', 'position'], axis=1)
        # y is the target variable
        y_train = df_train['actual_rating']
        # qid is the grouping parameter
        qid_train = df_train['srch_id']


        x_test = df_test.drop(['date_time', 'srch_id'], axis=1)
        qid_test = df_test['srch_id']

        print('x train:')
        print(x_train.columns)
        print('')
        print('x test')
        print(x_test.columns)

        # assume we already did hyperparameter tuning
        n = 100
        min_samples_leafs = 50
        max_depth = 10
        learning_rate = 0.1
        print('code needed for using the test set to predict for kaggle comp')

        metric = pyltr.metrics.NDCG(k=5)

        model = pyltr.models.LambdaMART(
                            metric=metric,
                            n_estimators = n,
                            min_samples_leaf = min_samples_leafs,
                            max_depth = max_depth,
                            learning_rate= learning_rate,
                            verbose = 1,
                            random_state=42)

        print('fit model')
        model.fit(x_train, y_train, qid_train)

        print('now predicting')
        y_pred = model.predict(x_test)

        print('converting to csv..')
        # Saving model output to 
        qid_np = qid_test.to_numpy()
        y_test_np = qid_test.to_numpy()
        # To obtain original prop_id for df_comp
        prop_df = df_test['prop_id']
        prop_np = prop_df.to_numpy()


        df_comp = pd.DataFrame({'srch_id':qid_np, 'y_test':y_test_np, 'y_pred':y_pred, 'prop_id':prop_np})

        print('df comp:')
        print(df_comp)

        # Assuming your dataframe is called 'df'
        sorted_df = df_comp.groupby('srch_id').apply(lambda x: x.sort_values('y_pred', ascending=False))

        # Reset the index of the sorted dataframe
        sorted_df = sorted_df.reset_index(drop=True)
        print(sorted_df)

        kaggle_df = sorted_df[['srch_id', 'prop_id']]
        print(kaggle_df)
        kaggle_df.to_csv('lambdaMart_group_118.csv', index=False)