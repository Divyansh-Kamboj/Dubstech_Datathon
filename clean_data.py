import pandas as pd

# Load the dataset
df = pd.read_csv("Access_to_Care_Dataset.csv")

# ── 1. Filtering ──

# Remove rows where FLAG is not null (unreliable estimates)
df = df[df["FLAG"].isnull()]

# Ensure ESTIMATE is numeric (coerce errors to NaN, then drop them)
df["ESTIMATE"] = pd.to_numeric(df["ESTIMATE"], errors="coerce")
df = df.dropna(subset=["ESTIMATE"])

# Keep only rows where GROUP is the specific age grouping
df = df[df["GROUP"] == "Age groups with 75 years and older"]

# ── 2. Department Mapping ──

def map_department(topic):
    topic_upper = topic.upper()
    if any(kw in topic_upper for kw in ["HEART", "CORONARY", "HYPERTENSION", "ANGINA", "INFARCTION"]):
        return "Cardiology"
    elif "CANCER" in topic_upper:
        return "Oncology"
    elif "DIABETES" in topic_upper:
        return "Endocrinology"
    elif any(kw in topic_upper for kw in ["DEPRESSION", "ANXIETY", "COUNSELED", "MENTAL HEALTH"]):
        return "Mental_Health"
    else:
        return "Drop"

df["Department"] = df["TOPIC"].apply(map_department)

# Drop rows that didn't match any department
df = df[df["Department"] != "Drop"]

# ── 3. Validation ──

print("First 5 rows of the cleaned dataframe:")
print(df.head())

print("\nUnique Department values:")
print(df["Department"].unique())

print("\nUnique SUBGROUP values (age groups):")
print(df["SUBGROUP"].unique())

# ── 4. Risk Score Calculation ──

DEPT_SEVERITY = {
    "Cardiology": 4.0,
    "Oncology": 5.0,
    "Mental_Health": 3.0,
    "Endocrinology": 2.0,
}

AGE_VULNERABILITY = {
    "75 years and older": 3.0,
    "65-74 years": 2.0,
    "45-64 years": 1.0,
}


def calculate_risk_score(row):
    score = 1.0
    score += DEPT_SEVERITY.get(row["Department"], 0.0)
    score += AGE_VULNERABILITY.get(row["SUBGROUP"], 0.0)
    return min(score, 10.0)


df["Risk_Score"] = df.apply(calculate_risk_score, axis=1)

# ── 5. Risk Score Validation ──

print("\nRisk Score check (Department, SUBGROUP, Risk_Score):")
print(df[["Department", "SUBGROUP", "Risk_Score"]].drop_duplicates().to_string(index=False))
