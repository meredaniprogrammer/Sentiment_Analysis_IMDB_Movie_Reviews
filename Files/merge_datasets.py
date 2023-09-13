import pandas as pd
import os
from convert_ratings import convert_ratings_to_out_of_10


def merge_datasets(dataset1_path, dataset2_path, output_path):
    # Check if the output file already exists
    if os.path.exists(output_path):
        print('Output file already exists. Merging datasets will be skipped.')
    else:
        # Load the datasets
        dataset1 = pd.read_csv(dataset1_path)
        dataset2 = pd.read_csv(dataset2_path)

        # Merge the datasets based on the 'id' column
        merged_data = pd.merge(dataset1, dataset2, on='id')

        # Replace 'rotten' with 'Negative' and 'fresh' with 'Positive' in the reviewState column
        merged_data['reviewState'] = merged_data['reviewState'].replace({'rotten': 'Negative', 'fresh': 'Positive'})

        # Select the required columns and rename them
        new_data = merged_data[['title', 'reviewText', 'criticName', 'originalScore', 'genre', 'reviewState']]
        new_data.columns = ['Title', 'Review', 'Reviewer', 'Rating', 'Genre', 'Sentiment']

        # Drop rows where the 'Review' is either empty or contains less than five words
        new_data = new_data[new_data['Review'].apply(lambda x: isinstance(x, str) and len(x.split()) >= 5)]

        # Convert ratings to be out of 5
        new_data = convert_ratings_to_out_of_10(new_data, 'Rating')

        # Save the new dataset into a csv file
        new_data.to_csv(output_path, index=False)

    print('Proceeding to the next step...')
