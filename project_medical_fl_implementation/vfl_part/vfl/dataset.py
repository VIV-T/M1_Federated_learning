from ucimlrepo import fetch_ucirepo
import pandas as pd

def load_dataset():

    dataset = fetch_ucirepo(id=891)

    X = dataset.data.features
    y = dataset.data.targets

    df = pd.concat([X,y],axis=1)

    return df