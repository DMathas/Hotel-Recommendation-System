import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

def Descriptives(df: pd.DataFrame) -> pd.DataFrame:
    """Generates a dataframe with several descriptive statistics."""

    cols = df.columns

    index = ['count', 'mu', 'sigma', 'kur', 'skew',  'JB', 'min', 'max']
    df_stats = pd.DataFrame(index)

    for i in cols:
        count = len(df[i])
        mu = float(np.mean(df[i]))
        sigma = float(np.std(df[i]))
        kur = float(stats.kurtosis(df[i]))
        skew = float(stats.skew(df[i]))
        JB = stats.jarque_bera(df[i])[1]
        minimum = float(np.min(df[i]))
        maximum = float(np.max(df[i]))
        new_df = np.around(pd.DataFrame([count, mu, sigma, kur, skew, JB, minimum, maximum ]), 3)
        df_stats = pd.concat([df_stats, new_df], axis= 1)
    
    df_stats.index = index
    df_stats = df_stats.iloc[:, 1:]
    df_stats.columns = cols
    #Round the values to three decimal places
    df_stats = df_stats.astype(float).round(3)

    return df_stats

def feat_eng_plot(data: pd.DataFrame) -> plt:
    """Produces a boxplot showing the distribution of the feature engineered colummns."""

    #Create a subset of the data containing the feature engineered variables
    subset_data = data[['prop_star_review_diff', 'price_diff', 'same_country_srch',
                        'mean_price_usd_sid', 'mean_prop_1_sid', 'mean_prop_2_sid',
                        'mean_prop_review_sid', 'mean_prop_star_sid', 'mean_price_usd_pid',
                        'mean_prop_1_pid', 'mean_prop_2_pid', 'mean_prop_review_pid',
                        'mean_prop_star_pid', 'mean_price_usd_pcid', 'mean_prop_1_pcid',
                        'mean_prop_2_pcid', 'mean_prop_review_pcid', 'mean_prop_star_pcid', 'position_proxy', 'click_proxy']]

    #Create two subplots in a 2x1 grid with adjusted spacing
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [1.4, 1]})

    # Plotting the upper box plot without log transformation
    sns.boxplot(data=subset_data, ax=axes[0], fliersize=0.1, linewidth=0.5)
    axes[0].set_xticklabels([])  # Remove x-axis ticks
    axes[0].set_xlabel('')  # Remove x-axis label
    axes[0].set_ylabel('Value', fontsize=7)
    axes[0].set_title('Distribution of Feature Engineered Variables (Without Price Variables Log Transformation)', fontsize=10)
    axes[0].grid(linestyle='dashed', linewidth=0.5)  # Add dashed gridlines

    #Apply logarithmic transformation to "price_usd" related columns
    price_columns = ['price_diff', 'mean_price_usd_sid', 'mean_price_usd_pid', 'mean_price_usd_pcid']
    subset_data[price_columns] = np.log1p(subset_data[price_columns])

    #Plotting the lower box plot with log transformation
    sns.boxplot(data=subset_data, ax=axes[1], fliersize=0.1, linewidth=0.5)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, fontsize=5)
    axes[1].set_xlabel('Feature', fontsize=7)
    axes[1].set_ylabel('Value', fontsize=7)
    axes[1].set_title('Distribution of Feature Engineered Variables (With Price Variables Log Transformation)', fontsize=10)
    axes[1].grid(linestyle='dashed', linewidth=0.5)  # Add dashed gridlines

    #Adjust the spacing between subplots
    plt.tight_layout()

    #Display the plot
    plt.show()


def hotel_price_distr(train_clean: pd.DataFrame):
    '''Distribution of hotel prices in the cleaned dataset, visualized by a bar plot with density line'''
    
    # Create a subset of the dataframe for the 'price_usd' column
    price_data = train_clean['price_usd']


    fig, ax = plt.subplots(figsize=(10, 6))
    # Create the combined plot with both histogram bars and KDE curve
    sns.histplot(data=price_data, bins=20, kde=True, line_kws={'linewidth': 2}, ax=ax)


    ax.set_xlabel('Price (USD)')
    ax.set_ylabel('Count/Density')
    ax.set_title('Distribution of Hotel Prices')
    plt.show()


def countrybook(train_clean: pd.DataFrame):
    '''Plot showing the frequency of users booking domestically or internationally. This is visualized on the clean train data set.'''

    # Count the number of overlapping values
    overlap_count = len(train_clean[(train_clean['visitor_location_country_id'].notna()) & (train_clean['prop_country_id'].notna()) & (train_clean['visitor_location_country_id'] == train_clean['prop_country_id'])])

    # Count the number of rows with different values
    different_count = len(train_clean[(train_clean['visitor_location_country_id'].notna()) & (train_clean['prop_country_id'].notna()) & (train_clean['visitor_location_country_id'] != train_clean['prop_country_id'])])

    # Count the number of rows with missing values in either column
    missing_count = len(train_clean[(train_clean['visitor_location_country_id'].isna()) | (train_clean['prop_country_id'].isna())])

    # Create a dataframe to store the counts
    data = {'Category': ['In country search', 'Out of country search', 'No information'],
            'Count': [overlap_count, different_count, missing_count]}
    counts_df = pd.DataFrame(data)

    # Plot
    sns.barplot(x='Category', y='Count', data=counts_df)
    plt.xlabel('Category')
    plt.ylabel('Frequency')
    plt.title('Domestic and International hotel search')
    plt.show()

def srch_page_position(raw_train: pd.DataFrame):
    '''Plot illustrating the booking and clicking frequency of hotels with respect to their page position.
    These score are normalized in order to fit neatly in one plot.'''

    # Calculate the total count for each position
    position_counts = raw_train['position'].value_counts()

    # Calculate the normalized frequencies for click_bool and booking_bool
    click_freq = raw_train.groupby('position')['click_bool'].sum() / position_counts
    booking_freq = raw_train.groupby('position')['booking_bool'].sum() / position_counts

    # Create a new DataFrame with the frequencies
    frequency_data = pd.DataFrame({'position': position_counts.index, 'click_freq': click_freq, 'booking_freq': booking_freq})

    # Melt dataframe for the countplot
    melted_data = frequency_data.melt(id_vars='position', var_name='bool_type', value_name='frequency')

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.set_context("paper")

    sns.barplot(x='position', y='frequency', hue='bool_type', data=melted_data)

    plt.title('Amount of bookings and clicks sorted per position')
    plt.xlabel('Position in page search')
    plt.ylabel('Normalized count')      # Normalized to have a plot that is more readable 
    plt.show()

def promotion_effect(raw_train: pd.DataFrame):
    '''Illustrates the effect of price promotion of a hotel, given in percentages. This is done both for click_bool and booking_bool'''

    # Calculate the percentages
    percentage_booking_promotion_0 = (raw_train.loc[(raw_train['promotion_flag'] == 0) & (raw_train['booking_bool'] == 1)].shape[0] / raw_train.loc[raw_train['promotion_flag'] == 0].shape[0]) * 100
    percentage_click_promotion_0 = (raw_train.loc[(raw_train['promotion_flag'] == 0) & (raw_train['click_bool'] == 1)].shape[0] / raw_train.loc[raw_train['promotion_flag'] == 0].shape[0]) * 100
    percentage_booking_promotion_1 = (raw_train.loc[(raw_train['promotion_flag'] == 1) & (raw_train['booking_bool'] == 1)].shape[0] / raw_train.loc[raw_train['promotion_flag'] == 1].shape[0]) * 100
    percentage_click_promotion_1 = (raw_train.loc[(raw_train['promotion_flag'] == 1) & (raw_train['click_bool'] == 1)].shape[0] / raw_train.loc[raw_train['promotion_flag'] == 1].shape[0]) * 100

    # Create a DataFrame for the percentages
    data = {
        'promotion_flag': ['0', '1'],
        'click dummy': [percentage_click_promotion_0, percentage_click_promotion_1],  # Updated label for click_bool
        'book dummy': [percentage_booking_promotion_0, percentage_booking_promotion_1]  # Updated label for booking_bool
    }
    df = pd.DataFrame(data)

    # Melt dataframe for the bar plot
    df_melt = pd.melt(df, id_vars='promotion_flag', var_name='bool_type', value_name='percentage')


    # Plotting
    colors = {'click dummy': '#FF8C00', 'book dummy': '#1F497D'}  # Updated legend labels
    sns.barplot(x='promotion_flag', y='percentage', hue='bool_type', data=df_melt, palette=colors)
    plt.title('Effect of price promotion')
    plt.xlabel('Promotion Flag')
    plt.ylabel('Percentage')
    plt.legend(title=None)
    plt.show()




