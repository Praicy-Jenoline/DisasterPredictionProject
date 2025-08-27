import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Paths
input_path = "data/processed/combined_disaster_data.csv"
output_path = "data/processed/combined_disaster_data_balanced.csv"

# Load dataset
df = pd.read_csv(input_path)

# Step 1: Convert one-hot encoded labels into single column
label_columns = ['is_earthquake', 'is_flood', 'is_landslide', 'is_cyclone']

def get_disaster_type(row):
    if row['is_earthquake'] == 1:
        return "earthquake"
    elif row['is_flood'] == 1:
        return "flood"
    elif row['is_landslide'] == 1:
        return "landslide"
    elif row['is_cyclone'] == 1:
        return "cyclone"
    else:
        return "none"

df['disaster_type'] = df.apply(get_disaster_type, axis=1)
df = df.drop(columns=label_columns)

print("Initial distribution:\n", df['disaster_type'].value_counts())

# Step 2: Seed underrepresented classes to a safe minimum
def seed_class(df, label, min_samples=6):
    current = df[df['disaster_type'] == label]
    if current.shape[0] < min_samples:
        template = current.iloc[0] if current.shape[0] > 0 else df.iloc[0]
        needed = min_samples - current.shape[0]
        new_rows = pd.DataFrame([template] * needed)
        new_rows['disaster_type'] = label
        df = pd.concat([df, new_rows], ignore_index=True)
    return df

# Make sure each class has at least 6 rows
for label in ["cyclone", "landslide", "none"]:
    df = seed_class(df, label, min_samples=6)

print("After seeding:\n", df['disaster_type'].value_counts())

# Step 3: Features/labels
X = df.drop(columns=["disaster_type"])
y = df["disaster_type"]

# Keep only numeric features (drop dates/strings)
X_numeric = X.select_dtypes(include=["int64", "float64"])

# Step 4: Choose safe k_neighbors based on smallest class size
min_class_size = y.value_counts().min()
k_neighbors = min(3, max(1, min_class_size - 1))

print(f"Using SMOTE with k_neighbors={k_neighbors}")

# Step 5: Apply SMOTE BEFORE train/test split
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_res, y_res = smote.fit_resample(X_numeric, y)

print("Balanced distribution:\n", pd.Series(y_res).value_counts())

# Step 6: Train/test split on balanced data
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, stratify=y_res, random_state=42
)

# Step 7: Save balanced dataset
balanced_df = pd.concat([pd.DataFrame(X_res), pd.DataFrame(y_res)], axis=1)
balanced_df.rename(columns={balanced_df.columns[-1]: "disaster_type"}, inplace=True)
balanced_df.to_csv(output_path, index=False)

print(f"\n✅ Balanced dataset saved to {output_path}")



