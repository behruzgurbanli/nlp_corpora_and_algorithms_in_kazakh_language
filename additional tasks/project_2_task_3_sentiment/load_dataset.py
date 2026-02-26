# load_dataset.py
import pandas as pd
import re

def load_imdb_dataset(csv_path="IMDB Dataset.csv", save_cleaned=True):
    """
    Load IMDb CSV, clean text, convert labels, and optionally save cleaned dataset.
    Returns a DataFrame with 'review' and 'sentiment' columns.
    """
    # Load CSV
    try:
        df = pd.read_csv(csv_path, names=["review","sentiment"], header=0)
    except:
        df = pd.read_csv(csv_path, names=["review","sentiment"], header=0, sep=';')

    # Drop missing rows
    df.dropna(inplace=True)

    # Strip spaces
    df["review"] = df["review"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.strip()

    # Convert labels to 0/1
    df["sentiment"] = df["sentiment"].map({"positive":1,"negative":0})

    # Clean text: lowercase, remove HTML tags, remove non-letters
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[^a-z\s]", "", text)
        return text

    df["review"] = df["review"].apply(clean_text)

    # Save cleaned dataset
    if save_cleaned:
        df.to_csv("IMDB_Cleaned_Dataset.csv", index=False)
        print("✅ Cleaned dataset saved as IMDB_Cleaned_Dataset.csv")

    return df

# If run directly
if __name__ == "__main__":
    df = load_imdb_dataset()
    print(df.head())
    print(f"Dataset shape: {df.shape}")