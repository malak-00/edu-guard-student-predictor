import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

RANDOM_STATE = 42

df = pd.read_csv("train.csv")
df.drop_duplicates(inplace=True)

num_cols = df.select_dtypes(include=["int64", "float64"]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

inc_cols = ["Gender", "International", "Scholarship holder", "Educational special needs"]
for column in inc_cols:
    df[column] = df[column].str.strip().str.lower()
    df[column] = df[column].replace({"y": "yes", "n": "no", "true": "yes", "false": "no"})
df["Gender"] = df["Gender"].replace({"f": "female", "m": "male"})

exclude_from_capping = {
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (without evaluations)",
}
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
for col in num_cols:
    if col in exclude_from_capping:
        continue
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])

target_order = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
df["Target"] = df["Target"].map(target_order)

strip_cols = ["Scholarship holder", "International", "Tuition fees up to date", "Debtor",
              "Educational special needs", "Displaced", "Daytime/evening attendance", "Gender"]
for col in strip_cols:
    df[col] = df[col].str.lower()

df["Gender"] = df["Gender"].map({"female": 1, "male": 0})
df["Daytime/evening attendance"] = df["Daytime/evening attendance"].map({"daytime": 1, "evening": 0})
df["Displaced"] = df["Displaced"].map({"yes": 1, "no": 0})
df["Educational special needs"] = df["Educational special needs"].map({"yes": 1, "no": 0})
df["Debtor"] = df["Debtor"].map({"yes": 1, "no": 0})
df["Tuition fees up to date"] = df["Tuition fees up to date"].map({"yes": 1, "no": 0})
df["International"] = df["International"].map({"yes": 1, "no": 0})
df["Scholarship holder"] = df["Scholarship holder"].map({"yes": 1, "no": 0})

remaining_cat_cols = df.select_dtypes(include="object").columns
remaining_cat_cols = remaining_cat_cols.drop(["Student_ID", "Application_ID", "Registration_Code"])
df_encoded = pd.get_dummies(df, columns=remaining_cat_cols, drop_first=True)

X = df_encoded.drop(columns=["Registration_Code", "Application_ID", "Student_ID", "Target"])
y = df_encoded["Target"]

feature_columns = X.columns.tolist()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
rf.fit(X_train_scaled, y_train)

y_pred = rf.predict(X_val_scaled)
acc = accuracy_score(y_val, y_pred)
f1_macro = f1_score(y_val, y_pred, average="macro")
report = classification_report(y_val, y_pred, target_names=list(target_order.keys()))

print(f"Validation accuracy: {acc:.4f}")
print(f"Validation macro-F1: {f1_macro:.4f}")
print(report)

X_all_scaled = scaler.fit_transform(X)
rf_final = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
rf_final.fit(X_all_scaled, y)

joblib.dump(rf_final, "model.joblib")
joblib.dump(scaler, "scaler.joblib")

raw_df = pd.read_csv("train.csv")

raw_num_cols = [c for c in raw_df.select_dtypes(include=["int64", "float64"]).columns if c != "Target"]
raw_cat_cols_for_form = [c for c in ["Marital status", "Application mode", "Course",
                                     "Previous qualification", "Nacionality",
                                     "Mother's qualification", "Father's qualification",
                                     "Mother's occupation", "Father's occupation"]]

schema = {
    "feature_columns": feature_columns,
    "target_order": target_order,
    "target_labels_by_index": {v: k for k, v in target_order.items()},
    "numeric_columns": raw_num_cols,
    "numeric_ranges": {
        c: [float(raw_df[c].min()), float(raw_df[c].max()), float(raw_df[c].median())]
        for c in raw_num_cols
    },
    "categorical_dropdown_columns": raw_cat_cols_for_form,
    "categorical_choices": {
        c: sorted(raw_df[c].dropna().unique().tolist()) for c in raw_cat_cols_for_form
    },
    "binary_columns": ["Gender", "Daytime/evening attendance", "Displaced",
                        "Educational special needs", "Debtor", "Tuition fees up to date",
                        "International", "Scholarship holder"],
    "one_hot_source_columns": remaining_cat_cols.tolist(),
    "validation_accuracy": acc,
    "validation_macro_f1": f1_macro,
}

with open("schema.json", "w") as f:
    json.dump(schema, f, indent=2)

print("\nSaved model.joblib, scaler.joblib, schema.json")
print("Feature count:", len(feature_columns))
