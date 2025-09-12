from imblearn.over_sampling import SMOTE
import pandas as pd

# Primer: tvoj dataset
# Zameni ovo sa svojim datasetom

df = pd.read_csv("heart_2020_cleaned.csv")

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Definiši izlazni CSV fajl i random_state
output_path = "balanced_dataset.csv"
random_state = 42

# Napravi SMOTE objekt i balansiraj dataset
smote = SMOTE(random_state=random_state)
X_res, y_res = smote.fit_resample(X, y)

# Napravi novi DataFrame
balanced_df = pd.DataFrame(X_res, columns=X.columns)
balanced_df["target"] = y_res

# Sačuvaj u CSV
balanced_df.to_csv(output_path, index=False)

print(f"Balansirani dataset sačuvan u '{output_path}' ({len(balanced_df)} redova)")
