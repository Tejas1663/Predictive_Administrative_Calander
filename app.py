from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import pandas as pd
import joblib
from catboost import CatBoostClassifier, Pool
import random

app = Flask(__name__)
app.secret_key = "your_secret_key"

# -------------------------------
# Load ML model and encoders
# -------------------------------
cat_model = CatBoostClassifier()
cat_model.load_model("catboost_event_model.cbm")
le_location = joblib.load("location_encoder.pkl")
le_event = joblib.load("event_encoder.pkl")
locations = le_location.classes_.tolist()

# -------------------------------
# Event Info
# -------------------------------
event_info = {
    "accident": {
        "recommendation": [
            "Check for injuries and call emergency services immediately.",
            "Move to a safe location if possible.",
            "Do not block traffic unnecessarily.",
            "Provide first aid if trained.",
            "Document the incident for authorities."
        ],
        "emergency_number": "108",
        "severity": "High",
        "precaution": "Always follow traffic rules and wear safety gear."
    },
    "clash": {
        "recommendation": [
            "Avoid the area immediately.",
            "Stay calm and do not engage.",
            "Inform local authorities about the situation.",
            "Seek shelter in a safe location.",
            "Keep communication lines open with family and friends."
        ],
        "emergency_number": "100",
        "severity": "Medium",
        "precaution": "Avoid large gatherings and protests."
    },
    "covid-19": {
        "recommendation": [
            "Wear a mask and maintain social distancing.",
            "Wash hands frequently with sanitizer.",
            "Stay home if feeling unwell.",
            "Get vaccinated and follow government guidelines.",
            "Inform local health authorities if symptoms appear."
        ],
        "emergency_number": "104",
        "severity": "High",
        "precaution": "Follow health advisories and quarantine rules."
    },
    "cyclone": {
        "recommendation": [
            "Seek shelter indoors and secure loose objects.",
            "Stock food, water, and emergency supplies.",
            "Avoid traveling during heavy winds.",
            "Stay tuned to local alerts and warnings.",
            "Have an evacuation plan ready."
        ],
        "emergency_number": "108",
        "severity": "High",
        "precaution": "Avoid low-lying areas and coastal regions."
    },
    "dengue": {
        "recommendation": [
            "Prevent mosquito breeding around your home.",
            "Use mosquito repellents and mosquito nets.",
            "Consult a doctor if fever or symptoms appear.",
            "Stay hydrated and rest adequately.",
            "Report suspected cases to health authorities."
        ],
        "emergency_number": "104",
        "severity": "Medium",
        "precaution": "Wear full sleeves and remove standing water."
    },
    "earthquake": {
        "recommendation": [
            "Drop, cover, and hold on immediately.",
            "Stay away from windows and heavy objects.",
            "If outside, move to an open area away from buildings.",
            "Keep communication lines clear.",
            "Check for injuries and damage after shaking stops."
        ],
        "emergency_number": "101",
        "severity": "High",
        "precaution": "Prepare an emergency kit and safe spots in your home."
    },
    "epidemic": {
        "recommendation": [
            "Follow local health authority guidelines.",
            "Maintain hygiene and avoid crowded areas.",
            "Seek medical advice if symptomatic.",
            "Stay home if unwell and avoid spreading infection.",
            "Report suspected cases to authorities."
        ],
        "emergency_number": "104",
        "severity": "High",
        "precaution": "Vaccinate if available and avoid unnecessary travel."
    },
    "health emergency": {
        "recommendation": [
            "Call emergency services immediately.",
            "Provide first aid if trained.",
            "Keep the patient calm and safe.",
            "Monitor vital signs until help arrives.",
            "Follow instructions from emergency responders."
        ],
        "emergency_number": "108",
        "severity": "High",
        "precaution": "Learn basic first aid and CPR."
    },
    "landslide": {
        "recommendation": [
            "Move to higher ground immediately.",
            "Avoid river valleys and low areas.",
            "Stay alert for warnings from authorities.",
            "Do not drive through debris or flooded areas.",
            "Help others if safe to do so."
        ],
        "emergency_number": "108",
        "severity": "High",
        "precaution": "Avoid construction near slopes and monitor heavy rainfall alerts."
    },
    "malaria": {
        "recommendation": [
            "Use mosquito repellents and nets.",
            "Avoid stagnant water to prevent breeding.",
            "Consult a doctor if fever appears.",
            "Take prescribed antimalarial medication if required.",
            "Report cases to local health authorities."
        ],
        "emergency_number": "104",
        "severity": "Medium",
        "precaution": "Wear protective clothing and use insecticide-treated nets."
    },
    "no event": {
        "recommendation": [
            "No specific action required.",
            "Stay alert to local news and weather updates.",
            "Maintain general safety precautions.",
            "Keep emergency contacts handy.",
            "Be prepared for any unexpected incidents."
        ],
        "emergency_number": "N/A",
        "severity": "Low",
        "precaution": "Maintain general safety habits."
    },
    "outbreak": {
        "recommendation": [
            "Follow health authority instructions carefully.",
            "Maintain hygiene and avoid crowded areas.",
            "Stay home if unwell.",
            "Report symptoms and suspected cases.",
            "Keep informed about outbreak updates."
        ],
        "emergency_number": "104",
        "severity": "High",
        "precaution": "Get vaccinated if applicable and avoid high-risk areas."
    },
    "power outage": {
        "recommendation": [
            "Use emergency lights or candles safely.",
            "Keep mobile devices charged.",
            "Avoid opening refrigerators frequently.",
            "Report the outage to local utility providers.",
            "Stay calm and inform neighbors if needed."
        ],
        "emergency_number": "1912",
        "severity": "Medium",
        "precaution": "Keep backup power and emergency kits ready."
    },
    "protest": {
        "recommendation": [
            "Avoid participating if unsafe.",
            "Stay away from the main protest areas.",
            "Follow updates from authorities.",
            "Keep communication open with family/friends.",
            "Seek shelter if the situation escalates."
        ],
        "emergency_number": "100",
        "severity": "Medium",
        "precaution": "Stay informed about protest plans and traffic diversions."
    },
    "riot": {
        "recommendation": [
            "Move to a safe location immediately.",
            "Do not engage with rioters.",
            "Call authorities if threatened.",
            "Stay indoors until safe.",
            "Help others if safe to do so."
        ],
        "emergency_number": "100",
        "severity": "High",
        "precaution": "Avoid riot-prone areas and stay alert to news alerts."
    },
    "social unrest": {
        "recommendation": [
            "Avoid the affected area.",
            "Stay calm and follow official instructions.",
            "Keep emergency contacts handy.",
            "Report unsafe situations to authorities.",
            "Assist vulnerable people safely if possible."
        ],
        "emergency_number": "100",
        "severity": "Medium",
        "precaution": "Stay updated via reliable news sources."
    },
    "traffic accident": {
        "recommendation": [
            "Call emergency services immediately.",
            "Move to a safe area if possible.",
            "Provide first aid if trained.",
            "Do not block traffic unnecessarily.",
            "Document the incident for authorities."
        ],
        "emergency_number": "108",
        "severity": "High",
        "precaution": "Follow traffic rules and drive safely."
    },
    "transport breakdown": {
        "recommendation": [
            "Move the vehicle to a safe location.",
            "Turn on hazard lights.",
            "Call towing or roadside assistance.",
            "Avoid standing in traffic lanes.",
            "Inform authorities if blocking roads."
        ],
        "emergency_number": "108",
        "severity": "Medium",
        "precaution": "Regularly maintain vehicles and check fuel and battery."
    },
    "tsunami": {
        "recommendation": [
            "Move to higher ground immediately.",
            "Stay away from the coast and low-lying areas.",
            "Follow tsunami alerts and official instructions.",
            "Keep emergency kits ready.",
            "Assist others if safe to do so."
        ],
        "emergency_number": "108",
        "severity": "High",
        "precaution": "Know evacuation routes if living in coastal areas."
    },
    "water shortage": {
        "recommendation": [
            "Use water sparingly and avoid wastage.",
            "Store enough drinking water safely.",
            "Report leaks or damages to local authorities.",
            "Follow water rationing rules if applicable.",
            "Recycle and reuse water when possible."
        ],
        "emergency_number": "181",
        "severity": "Medium",
        "precaution": "Conserve water and educate family/community about water-saving habits."
    }
}

# -------------------------------
# Database setup
# -------------------------------
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

init_db()

# -------------------------------
# Routes
# -------------------------------

# Home Page
@app.route("/")
def home():
    return render_template("home.html")

# Signup
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        try:
            conn = sqlite3.connect("users.db")
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            conn.close()
            flash("Signup successful! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists. Try another.", "danger")
    return render_template("signup.html")

# Login
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()
        conn.close()
        if user:
            session["username"] = username
            return redirect(url_for("index"))
        else:
            flash("Invalid credentials. Try again.", "danger")
    return render_template("login.html")

# Logout
@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("home"))

# Event Prediction (Protected)
@app.route("/index", methods=["GET", "POST"])
def index():
    if "username" not in session:
        return redirect(url_for("login"))

    prediction = None
    recommendation = []
    emergency_number = None
    severity = None
    precaution = None

    if request.method == "POST":
        date_str = request.form["date"]
        location = request.form["location"]
        year, month, day = map(int, date_str.split('-'))

        single_input = pd.DataFrame([{
            'Year': year,
            'Month': month,
            'Day': day,
            'Location': location
        }])

        single_input['Location'] = le_location.transform(single_input['Location'])
        input_pool = Pool(single_input, cat_features=[single_input.columns.get_loc('Location')])
        pred_encoded = cat_model.predict(input_pool)
        pred_event = le_event.inverse_transform([int(pred_encoded[0][0])])
        prediction = pred_event[0]

        if prediction in event_info:
            info = event_info[prediction]
            recommendation = info["recommendation"]
            emergency_number = info["emergency_number"]
            severity = info["severity"]
            precaution = info["precaution"]
        else:
            recommendation = ["No recommendation available."]
            emergency_number = "N/A"
            severity = "N/A"
            precaution = "N/A"

    return render_template(
        "index.html",
        prediction=prediction,
        recommendation=recommendation,
        emergency_number=emergency_number,
        severity=severity,
        precaution=precaution,
        locations=locations
    )

if __name__ == "__main__":
    app.run(debug=True)