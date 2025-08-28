import model_testing as model
import pandas as pd

# Load your dataset
df = pd.read_excel("empty_gender_name_liste.xlsx")

# Apply predict_gender safely
def safe_predict(name):
    if pd.isna(name):  # Skip empty cells
        return None
    name = str(name).strip()  # Ensure it's a string
    if not name:  # Skip if it's empty after stripping
        return None
    return model.predict_gender(name)

# Create new column with predictions
df["Gender"] = df["Name"].apply(safe_predict)

# Save result
df.to_excel("result.xlsx", index=False)
