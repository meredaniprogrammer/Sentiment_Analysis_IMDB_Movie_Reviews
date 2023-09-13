import pandas as pd
import os

def reduce_dataset(input_path, output_path):
    # Check if the output file already exists
    if os.path.exists(output_path):
        print('File already exists. The function will not run.')
        return

    # Load the dataset
    dataset = pd.read_csv(input_path)

    # Drop the 'Title', 'Reviewer' and 'Genre' columns
    dataset = dataset.drop(columns=['Title', 'Reviewer', 'Genre'])

    # Save the new dataset into a csv file
    dataset.to_csv(output_path, index=False)

