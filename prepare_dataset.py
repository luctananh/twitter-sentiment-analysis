"""
Script to prepare and sample dataset from Kaggle Sentiment140
Creates a subset of 300,000 random tweets for efficient model training
"""

import pandas as pd
import os
import kagglehub
from tqdm import tqdm
import numpy as np

def prepare_dataset(negative_count=150000, positive_count=150000, output_folder="kaggle_datasets"):
    """
    Download full dataset from Kaggle and create balanced samples
    
    Args:
        negative_count (int): Number of negative tweets to sample (default: 150,000)
        positive_count (int): Number of positive tweets to sample (default: 150,000)
        output_folder (str): Folder to save the sampled dataset
    """
    
    print("=" * 70)
    print("  📥 Sentiment140 Dataset Preparation Script (Balanced)")
    print("=" * 70)
    
    print(f"\n📥 Downloading Sentiment140 dataset from Kaggle...")
    try:
        dataset_path = kagglehub.dataset_download("kazanova/sentiment140")
        csv_file = os.path.join(dataset_path, "training.1600000.processed.noemoticon.csv")
    except Exception as e:
        print(f"❌ Error downloading from Kaggle: {e}")
        print("   Make sure you have kagglehub configured correctly")
        return False
    
    if not os.path.exists(csv_file):
        print(f"❌ Error: Dataset file not found at {csv_file}")
        return False
    
    print(f"✅ Dataset found at: {csv_file}")
    print(f"\n📖 Loading full dataset...")
    
    # Load full CSV
    try:
        df = pd.read_csv(csv_file, encoding='latin1', header=None)
        print(f"✅ Loaded {len(df):,} tweets from full dataset")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return False
    
    # Set column names
    df.columns = ["sentiment", "id", "date", "query", "user", "text"]
    
    # Replace sentiment values FIRST: 0 → -1, 4 → 1
    print("🔄 Converting sentiment values: 0 → -1 (negative), 4 → 1 (positive)")
    df["sentiment"] = df["sentiment"].replace({0: -1, 4: 1})
    
    # Separate by sentiment
    print(f"\n🔍 Filtering tweets by sentiment...")
    df_negative = df[df["sentiment"] == -1]
    df_positive = df[df["sentiment"] == 1]
    
    print(f"   📊 Negative tweets (-1): {len(df_negative):,}")
    print(f"   📊 Positive tweets (1): {len(df_positive):,}")
    
    # Sample from each sentiment
    print(f"\n🎲 Sampling {negative_count:,} negative tweets (seed=42)...")
    df_negative_sample = df_negative.sample(n=negative_count, random_state=42)
    
    print(f"🎲 Sampling {positive_count:,} positive tweets (seed=42)...")
    df_positive_sample = df_positive.sample(n=positive_count, random_state=42)
    
    # Combine samples
    print(f"\n🔗 Combining samples...")
    print(f"   📍 Negative (-1) first, then Positive (1)")
    df_sample = pd.concat([df_negative_sample, df_positive_sample], ignore_index=True)
    
    total_samples = len(df_sample)
    
    # Create output folder if not exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Save to CSV
    output_file = os.path.join(output_folder, f"sentiment140_sample_{total_samples}.csv")
    print(f"\n💾 Saving sampled dataset to {output_file}...")
    
    try:
        # Save without index and header (match original format)
        df_sample.to_csv(output_file, encoding='latin1', index=False, header=False)
        print(f"✅ File saved successfully!")
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")
        return False
    
    # Print statistics
    file_size_mb = os.path.getsize(output_file) / (1024**2)
    print(f"\n" + "=" * 70)
    print(f"  📊 Balanced Dataset Summary")
    print("=" * 70)
    print(f"✅ Total sample size: {total_samples:,} tweets")
    print(f"📁 Output file: {output_file}")
    print(f"💾 File size: {file_size_mb:.2f} MB")
    
    # Print sentiment distribution
    print(f"\n📈 Sentiment Distribution (BALANCED):")
    sentiment_counts = df_sample["sentiment"].value_counts().sort_index()
    
    sentiment_names = {-1: "Negative", 0: "Neutral", 1: "Positive"}
    for sentiment in [-1, 0, 1]:
        count = sentiment_counts.get(sentiment, 0)
        if count > 0:
            percentage = (count / total_samples) * 100
            bar = "█" * int(percentage / 2)
            print(f"   {sentiment_names[sentiment]} ({sentiment:2d}): {count:7,} tweets ({percentage:5.1f}%) {bar}")
    
    print(f"\n" + "=" * 70)
    print(f"✅ Balanced dataset preparation completed successfully!")
    print("=" * 70)
    print(f"\n💡 Next step: Run the training script")
    print(f"   cd model_TFIDF")
    print(f"   python LogisticRegression_TFIDF.py")
    print(f"\n")
    
    return True

def main():
    """Main function"""
    # Prepare balanced dataset: 150K negative + 150K positive = 300K total
    success = prepare_dataset(
        negative_count=150000,
        positive_count=150000,
        output_folder="kaggle_datasets"
    )
    
    if not success:
        print("\n❌ Dataset preparation failed!")
        return
    
    print("✅ Ready to train model on 300K tweets!")

if __name__ == "__main__":
    main()
