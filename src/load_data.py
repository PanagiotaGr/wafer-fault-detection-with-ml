import pandas as pd

def load_data(path="data/raw/LSWMD.pkl"):
    df = pd.read_pickle(path)
    print("Shape:", df.shape)
    print("Columns:", df.columns)
    print(df.head())
    return df

if __name__ == "__main__":
    df = load_data()
