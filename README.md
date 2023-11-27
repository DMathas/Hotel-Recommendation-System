# Hotel Recommendation System

This repository contains the code for a hotel recommendation system developed as part of a project. The system is built on a framework with various components, including data preprocessing, exploratory data analysis, and the implementation of different recommendation models.

## Authors
David Mathas, Adam Pohle & Can Efe Gursoy - June 2023

## Components

1. **Data Loading and Preprocessing**
   - The data loading and preprocessing are handled in the `preprocessing.py` module.
   - The `main.py` script includes functions for loading and cleaning the raw dataset.

2. **Exploratory Data Analysis**
   - The `explore.py` module performs exploratory data analysis, generating descriptive statistics and plots.

3. **Content-Based Filtering**
   - The `content_based_filtering.py` module implements a content-based filtering recommender system.

4. **Random Forest Model**
   - The `randomforest.py` module contains functions for training a random forest model and generating predictions.

5. **LambdaMart Model**
   - The `lambda_val.py` module includes the implementation of the LambdaMart model for hotel recommendation.

6. **Subset NDCG Calculation**
   - The `subset_ndcg.py` module handles subsetted testing and NDCG score calculations.

## Usage

To run the main program and execute the entire recommendation system, use the `main()` function in `main.py`. Adjust the parameters as needed, such as specifying whether the data is preprocessed or for hyperparameter training.

```python
if __name__ == '__main__':
    main()
```

## Additional Notes:

This project was part of the course Data Mining Techniques at the Vrije Universiteit Amsterdam (VU).
