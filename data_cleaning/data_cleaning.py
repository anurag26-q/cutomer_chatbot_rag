import pandas as pd

# File paths
file_paths = {
    "headphones": r"data/raw_data/headphones_data.csv",
    "mobiles": r"data/raw_data/mobiles_data.csv",
    "smart_watches": r"data/raw_data/smart_watches_data.csv",
    "tv": r"data/raw_data/tv_data.csv"
}

# More comprehensive invalid review values
invalid_reviews = {"n|a", "review not found", "", "no reviews found", "na", "n/a", "nan","none"}

# Clean and recheck
cleaned_dfs = {}
valid_counts = {}

for category, path in file_paths.items():
    df = pd.read_csv(path)
    df['Reviews'] = df['Reviews'].astype(str).str.strip().str.lower()
    df_cleaned = df[~df['Reviews'].isin(invalid_reviews)].copy()
    
    cleaned_dfs[category] = df_cleaned
    valid_counts[category] = len(df_cleaned)
    
    # Save cleaned version
    cleaned_path = f"data/processed_data/{category}_data_cleaned.csv"
    df_cleaned.to_csv(cleaned_path, index=False)

valid_counts
