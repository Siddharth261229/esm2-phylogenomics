import pandas as pd

meta = pd.read_csv("output/metadata.csv")

for fam in meta["family"].unique():
    s = set(meta[meta["family"] == fam]["species"])
    print(fam, len(s))

families = meta["family"].unique()

shared = None

for fam in families:
    s = set(meta[meta["family"] == fam]["species"])

    if shared is None:
        shared = s
    else:
        shared &= s

print("Shared species:", len(shared))