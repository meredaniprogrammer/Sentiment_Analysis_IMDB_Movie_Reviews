import pandas as pd

def convert_ratings_to_out_of_10(df, rating_column):
    # Drop rows with empty rating and genre
    df = df.dropna(subset=[rating_column, 'Genre'])

    # Keep rows where rating contains exactly one '/' and both parts are numeric
    df = df[df[rating_column].apply(
        lambda x: x.count('/') == 1 and all(part.replace('.', '', 1).isnumeric() for part in x.split('/')))]

    def process_rating(rating):
        score, total = rating.split('/')
        # Strip trailing periods and convert to float
        score, total = map(float, [score.rstrip('.'), total.rstrip('.')])
        # Check for zero division error
        if total == 0:
            return 0
        # Return the score scaled to out of 10 and round to nearest integer
        return round((10 * score) / total)

    df[rating_column] = df[rating_column].apply(process_rating)

    return df
