import pandas as pd
import os

def clean_movie_data(filepath='movies_data.csv'):
    df = pd.read_csv(filepath)

    # Remove irrelevant columns
    df.drop(columns=['id', 'imdb_id', 'homepage', 'tagline'], inplace=True)

    # Drop rows with critical missing values
    df.dropna(subset=['original_title', 'budget', 'revenue', 'runtime', 'genres'], inplace=True)

    # Convert release date
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df = df[df['release_date'].notna()]

    # Cast to numeric
    df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')

    # Create new features
    df['profit'] = df['revenue'] - df['budget']
    df['revenue_category'] = pd.cut(df['revenue'],
                                    bins=[0, 1e6, 1e7, 1e8, df['revenue'].max()],
                                    labels=['Low', 'Medium', 'High', 'Blockbuster'])

    # Filter out unrealistic entries
    df = df[(df['budget'] > 0) & (df['revenue'] > 0) & (df['runtime'] > 0)]

    # Save cleaned data
    output_path = "clean_movies_data.csv"
    df.to_csv(output_path, index=False)
    
    print(f"[✔] File saved as: {os.path.abspath(output_path)}")
    return df

# Run this if file is executed directly
if __name__ == "__main__":
    clean_movie_data()



