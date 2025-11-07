# ===============================
# TRAIN CATBOOST MODEL FROM CSV
# ===============================
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# -------------------------------
# 1️⃣ Load CSV
# -------------------------------
df = pd.read_csv('predictive_calendar_60000.csv')  # replace with your CSV path
print(f"✅ Dataset loaded with {len(df)} rows")
print(df.head())

# -------------------------------
# 2️⃣ Prepare Features & Target
# -------------------------------
X = df[['Year','Month','Day','Location']]
y = df['Event']

# Encode categorical columns
le_location = LabelEncoder()
X['Location'] = le_location.fit_transform(X['Location'])

le_event = LabelEncoder()
y_encoded = le_event.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# -------------------------------
# 3️⃣ Train CatBoost Model
# -------------------------------
cat_features = [X.columns.get_loc('Location')]  # categorical column index
train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

cat_model = CatBoostClassifier(
    iterations=400,
    learning_rate=0.05,
    depth=8,
    loss_function='MultiClass',
    verbose=100
)
cat_model.fit(train_pool)

# -------------------------------
# 4️⃣ Predict & Evaluate
# -------------------------------
y_pred = cat_model.predict(X_test)
y_pred = [int(p[0]) for p in y_pred]

print(f"🎯 Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=le_event.classes_))

# -------------------------------
# 5️⃣ Save Model & Encoders
# -------------------------------
cat_model.save_model("catboost_event_model.cbm")
joblib.dump(le_location, "location_encoder.pkl")
joblib.dump(le_event, "event_encoder.pkl")
print("✅ CatBoost Model & Encoders Saved for Future Predictions")
