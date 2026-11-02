from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw" / "HDFS_v1" / "preprocessed"
PROCESSED_DIR = ROOT / "data" / "processed"

TRACE_FILE = RAW_DIR / "Event_traces.csv"
LABEL_FILE = RAW_DIR / "anomaly_label.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():

    traces = pd.read_csv(TRACE_FILE)

    labels = pd.read_csv(LABEL_FILE)

    dataset = traces.merge(
    labels,
    on="BlockId",
    how="inner")

    dataset = dataset.drop(
    columns=["Label_x"])

    dataset = dataset.rename(
    columns = {
        "Label_y": "Label" })

    print(dataset.columns)
    print(dataset.head())

    dataset["Text"] = (
    dataset["Features"]
        .astype(str)
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace(",", " ", regex=False)
)

    dataset["Label"] = dataset["Label"].map({
    "Normal": 0,
    "Anomaly": 1
})

    dataset = dataset[
    [
        "Text",
        "Label"
    ]
]
    return dataset


def save_dataset(df):

    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["Label"],
        random_state=42
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["Label"],
        random_state=42
    )

    df.to_csv(
        PROCESSED_DIR / "processed_dataset.csv",
        index=False
    )

    train_df.to_csv(
        PROCESSED_DIR / "train.csv",
        index=False
    )

    val_df.to_csv(
        PROCESSED_DIR / "val.csv",
        index=False
    )

    test_df.to_csv(
        PROCESSED_DIR / "test.csv",
        index=False
    )

    print("\nDataset Summary\n")

    print(df["Label"].value_counts())

    print()

    print(df["Label"].value_counts(normalize=True))

    print()

    print(f"Total Samples : {len(df)}")
    print(f"Train Samples : {len(train_df)}")
    print(f"Validation Samples : {len(val_df)}")
    print(f"Test Samples : {len(test_df)}")


if __name__ == "__main__":

    dataset = load_dataset()

    save_dataset(dataset)