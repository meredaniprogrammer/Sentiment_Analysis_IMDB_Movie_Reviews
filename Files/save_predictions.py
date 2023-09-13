# save_predictions.py
import pandas as pd
import os

def save_predictions(titles, reviews, y_test, predictions, output_path):
    # Check if the output file already exists
    if os.path.exists(output_path):
        print(f'File {output_path} already exists. The function will not run.')
        return

    # Create a DataFrame with the necessary data
    data = {
        'Title': titles,
        'Review': reviews,
        'Actual': y_test,
        'Predicted': predictions
    }
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(output_path, index=False)
