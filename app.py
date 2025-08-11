from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
import pandas as pd
import csv
import io
import pickle
import os
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, firestore, auth
from flask_mail import Mail, Message
import requests
import random
import string
import threading
import time

# Firebase initialization
cred = credentials.Certificate(r"D:\OLD D back\SEM 9\MINOR PROJECT - DECISION TOOL DEVELOPMENT\final\earthquake-warning-system_CAT2\climateinformed2024-firebase-adminsdk-pyly2-657bb402a5.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_key')  # Use environment variable or fallback

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'youremail@gmail.com'
app.config['MAIL_PASSWORD'] = 'your_pwd'
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False

mail = Mail(app)

def generate_random_data():
    while True:
        # Generate random earthquake data
        earthquake_data = {
            'features': [random.uniform(1.0, 10.0),  # Magnitude
                         random.uniform(0.0, 700.0), # Depth
                         random.uniform(-90.0, 90.0), # Latitude
                         random.uniform(-180.0, 180.0), # Longitude
                         random.uniform(0.0, 10.0),  # dmin
                         random.uniform(0.0, 360.0), # gap
                         random.randint(1, 100)],    # nst
            'prediction': random.randint(0, 3),  # Prediction (0: Green, 1: Yellow, 2: Orange, 3: Red)
            'date': datetime.utcnow().isoformat()
        }

        # Store the random earthquake data in Firebase
        db.collection('earthquake').add(earthquake_data)

        # Generate random flood data
        flood_data = {
            'features': [random.uniform(0.0, 100.0),  # Monsoon Intensity
                         random.uniform(0.0, 100.0),  # Urbanization
                         random.uniform(0.0, 100.0)], # Climate Change
            'prediction': random.uniform(0.0, 1.0),  # Prediction
            'date': datetime.utcnow().isoformat()
        }

        # Store the random flood data in Firebase
        db.collection('flood').add(flood_data)

        # Generate random tsunami data
        tsunami_data = {
            'features': [random.uniform(1.0, 10.0),  # EQ_MAGNITUDE
                         random.uniform(0.0, 700.0), # EQ_DEPTH
                         random.uniform(0.0, 10.0)], # TS_INTENSITY
            'prediction': random.randint(0, 3),      # Prediction (0: Green, 1: Yellow, 2: Orange, 3: Red)
            'date': datetime.utcnow().isoformat()
        }

        # Store the random tsunami data in Firebase
        db.collection('tsunami').add(tsunami_data)

        # Generate random tornado data
        tornado_data = {
            'features': [random.uniform(1.0, 10.0),  # Magnitude
                         random.choice(['TX', 'OK', 'KS', 'MO']),  # State
                         random.randint(0, 1000),   # Injuries
                         random.randint(0, 500)],   # Fatalities
            'prediction': random.randint(0, 3),     # Prediction (0: Green, 1: Yellow, 2: Orange, 3: Red)
            'date': datetime.utcnow().isoformat()
        }

        # Store the random tornado data in Firebase
        db.collection('tornado').add(tornado_data)

        # Wait for 5 seconds before generating new data
        time.sleep(5)

# Start the background task
data_generation_thread = threading.Thread(target=generate_random_data)
data_generation_thread.daemon = True
data_generation_thread.start()

# Load models and scalers
models = {
    'earthquake': {
        'model': pickle.load(open('earthquake_model.pkl', 'rb')),
        'scaler': pickle.load(open('earthquake_scaler.pkl', 'rb')),
    },
    'flood': {
        'model': pickle.load(open('flood_model.pkl', 'rb')),
        'scaler': pickle.load(open('flood_scaler.pkl', 'rb')),
    },
    'tornado': {
        'model': pickle.load(open('tornado_model.pkl', 'rb')),
        'scaler': pickle.load(open('tornado_scaler.pkl', 'rb')),
    },
    'tsunami': {
        'model': pickle.load(open('tsunami_model.pkl', 'rb')),
        'scaler': pickle.load(open('tsunami_scaler.pkl', 'rb')),
    }
}

# Load and process datasets
datasets = {
    'earthquake': pd.read_csv('earthquake.csv'),
    'flood': pd.read_csv('flood.csv'),
    'tornadoes': pd.read_csv('tornadoes.csv'),
    'tsunami': pd.read_csv('tsunami.csv')
}

# Function to get location from latitude and longitude
def get_location_from_latlng(lat, lon):
    api_key = '0dd36f23d63b4f70a020ffc55fe20783'  # Replace with your OpenCage API key
    url = f'https://api.opencagedata.com/geocode/v1/json?q={lat}+{lon}&key={api_key}'
    response = requests.get(url)
    data = response.json()
    
    if data['results']:
        location = data['results'][0]
        return location['formatted']
    else:
        return 'Location details not found.'



@app.route('/download_dataset', methods=['POST'])
def download_dataset():
    disaster_type = request.form.get('disaster_download_type')
    docs = db.collection(disaster_type).stream()
    
    # Create an in-memory CSV file
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    headers = {
        'earthquake': ['magnitude', 'depth', 'latitude', 'longitude', 'dmin', 'gap', 'nst', 'prediction'],
        'flood': ['MonsoonIntensity', 'Urbanization', 'ClimateChange', 'FloodProbability', 'prediction'],
        'tornado': ['magnitude', 'state', 'injuries', 'fatalities', 'prediction'],
        'tsunami': ['EQ_MAGNITUDE', 'EQ_DEPTH', 'TS_INTENSITY', 'prediction']
    }
    writer.writerow(headers.get(disaster_type, []))

    # Write data rows
    for doc in docs:
        data = doc.to_dict()
        row = data.get('features', []) + [data.get('prediction')]
        writer.writerow(row)

    # Create a CSV file to send to the user
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode('utf-8')), mimetype='text/csv',
                     as_attachment=True, download_name=f'{disaster_type}_dataset.csv')

@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('main_dashboard'))
    return render_template('login.html')

@app.route('/main_dashboard')
def main_dashboard():
    if 'user' in session:
        return render_template('dashboard.html')
    return redirect(url_for('login'))

@app.route('/send_alert_email', methods=['POST'])
def send_alert_email():
    alert_level = request.form.get('alert_level')
    disaster_details = request.form.get('disaster_details')
    user_email = session.get('user')

    if not user_email:
        flash('User is not logged in!', 'danger')
        return redirect(url_for('main_dashboard'))

    # Map alert level to CSS class
    alert_level_classes = {
        'Green': 'green',
        'Yellow': 'yellow',
        'Orange': 'orange',
        'Red': 'red'
    }
    alert_level_class = alert_level_classes.get(alert_level, 'green')  # Default to 'green'

    email_data = {
        'alert_level': alert_level,
        'alert_level_class': alert_level_class,
        'disaster_details': disaster_details
    }

    subject = f"Disaster Alert Level: {alert_level}"
    body = render_template('email_template.html', **email_data)
    
    msg = Message(subject, sender=app.config['MAIL_USERNAME'], recipients=[user_email])
    msg.html = body
    
    try:
        mail.send(msg)
        flash('Email sent successfully!', 'success')
    except Exception as e:
        flash(f'Failed to send email: {str(e)}', 'danger')
        app.logger.error(f'Failed to send email: {str(e)}')  # Log the error

    return redirect(url_for('main_dashboard'))

@app.route('/emergency_contacts', methods=['GET', 'POST'])
def emergency_contacts():
    if 'user' not in session:
        flash('You need to be logged in to manage emergency contacts.', 'danger')
        return redirect(url_for('login'))

    user_email = session['user']

    if request.method == 'POST':
        contact_email = request.form.get('contact_email')

        # Save the contact email to Firestore under the user's document
        db.collection('users').document(user_email).update({
            'emergency_contacts': firestore.ArrayUnion([contact_email])
        })

        flash('Emergency contact added successfully!', 'success')
        return redirect(url_for('emergency_contacts'))

    # Retrieve existing emergency contacts from Firestore
    doc = db.collection('users').document(user_email).get()
    contacts = []
    if doc.exists:
        user_data = doc.to_dict()
        contacts = user_data.get('emergency_contacts', [])

    return render_template('emergency_contacts.html', contacts=contacts)


@app.route('/predict_disaster', methods=['GET', 'POST'])
def predict_disaster():
    if request.method == 'POST':
        disaster_type = request.form['disaster_type']
        latitude = request.form['latitude']
        longitude = request.form['longitude']
        
        # Fetch location details
        location_details = get_location_from_latlng(latitude, longitude)

        # Define the required features for each disaster type
        feature_mapping = {
            'earthquake': ['magnitude', 'depth', 'latitude', 'longitude', 'dmin', 'gap', 'nst'],
            'flood': ['MonsoonIntensity', 'Urbanization', 'ClimateChange'],
            'tornado': ['magnitude', 'state', 'injuries', 'fatalities'],
            'tsunami': ['EQ_MAGNITUDE', 'EQ_DEPTH', 'TS_INTENSITY']
        }
        
        features = []
        for feature in feature_mapping[disaster_type]:
            features.append(float(request.form.get(feature, 0)))  # Default to 0 if feature is missing
        
        # Scale features and make prediction
        features_scaled = models[disaster_type]['scaler'].transform([features])
        prediction = models[disaster_type]['model'].predict(features_scaled)[0]

        # Convert prediction and features to standard Python types
        if disaster_type.lower() == 'earthquake':
            prediction = int(prediction)
        
        features_standard = [float(f) for f in features]

        # Store the input and result in Firestore
        current_date = datetime.utcnow()  # Get the current UTC date and time

        db.collection(disaster_type).add({
            'features': features_standard,
            'prediction': float(prediction),
            'date': current_date.isoformat()  # Convert datetime to ISO format string
        })

        # Prepare disaster details to be sent in the email
        disaster_details = (f"Disaster Type: {disaster_type.capitalize()}, Latitude: {latitude}, Longitude: {longitude}, "
                            f"Location: {location_details}, Prediction: {prediction}")

        # If prediction is "RED," send email to emergency contacts
        if prediction == 3:  # Assuming 3 corresponds to "RED"
            user_email = session.get('user')
            if user_email:
                # Retrieve emergency contacts from Firestore
                doc = db.collection('users').document(user_email).get()
                if doc.exists:
                    emergency_contacts = doc.to_dict().get('emergency_contacts', [])
                    
                    # Send an email to each emergency contact
                    for contact in emergency_contacts:
                        subject = f"Urgent Disaster Alert: {disaster_type.capitalize()} RED Alert"
                        body = render_template('email_template.html', alert_level="RED", disaster_details=disaster_details)
                        
                        msg = Message(subject, sender=app.config['MAIL_USERNAME'], recipients=[contact])
                        msg.html = body
                        
                        try:
                            mail.send(msg)
                            flash(f'Alert email sent to {contact}.', 'success')
                        except Exception as e:
                            flash(f'Failed to send email to {contact}: {str(e)}', 'danger')

        return render_template('result.html', disaster_type=disaster_type.capitalize(), prediction=prediction, disaster_details=disaster_details)
    return render_template('predict_disaster.html')



@app.route('/analyze_disasters', methods=['GET', 'POST'])
def analyze_disasters():
    # Initialize variables
    greenCnt = 0
    orangeCnt = 0
    yellowCnt = 0
    redCnt = 0
    magnitude = []
    tsunami = {}
    flood = {}  # Initialize flood as an empty dictionary
    disaster_type = ''
    dates=[]

    if request.method == 'POST':
        disaster_type = request.form.get('disaster_type')
        collection = db.collection(disaster_type)
        docs = collection.stream()
        records = [doc.to_dict() for doc in docs]

        if disaster_type == 'earthquake':
            for record in records:
                
                magnitude.append(float(record['features'][0]))  # Convert to float if necessary
                if record['prediction'] == 0:
                    greenCnt += 1
                elif record['prediction'] == 1:
                    yellowCnt += 1
                elif record['prediction'] == 2:
                    orangeCnt += 1
                else:
                    redCnt += 1

        elif disaster_type == 'tsunami':
            earthquake_magnitude = []
            earthquake_depth = []
            tsunami_intensity = []

            for record in records:
                earthquake_magnitude.append(float(record['features'][0]))  # Convert to float
                earthquake_depth.append(float(record['features'][1]))      # Convert to float
                tsunami_intensity.append(float(record['features'][2]))      # Convert to float
                if record['prediction'] == 0:
                    greenCnt += 1
                elif record['prediction'] == 1:
                    yellowCnt += 1
                elif record['prediction'] == 2:
                    orangeCnt += 1
                else:
                    redCnt += 1

            tsunami = {"Earthquake_Magnitude": earthquake_magnitude, "Earthquake_Depth": earthquake_depth, "Tsunami_Intensity": tsunami_intensity}

        elif disaster_type == 'flood':
            monsoonIntensity = []
            urbanizationLevel = []
            climateChange = []

            for record in records:
                monsoonIntensity.append(float(record['features'][0]))  # Convert to float
                urbanizationLevel.append(float(record['features'][1]))
                print(record['features'][2])  # Convert to float
                climateChange.append(float(record['features'][2]))      # Convert to float
                if record['prediction'] <= 0.4:
                    greenCnt += 1
                elif record['prediction'] > 0.4 and record['prediction'] <= 0.6:
                    yellowCnt += 1
                elif record['prediction'] > 0.6 and record['prediction'] <= 0.8:
                    orangeCnt += 1
                else:
                    redCnt += 1

            flood = {
                "Monsoon_Intensity": monsoonIntensity,
                "Urbanization_Level": urbanizationLevel,
                "Climate_Change": climateChange
            }

        data = {"Green": greenCnt, "Yellow": yellowCnt, "Orange": orangeCnt, "Red": redCnt}
          
        return render_template('analyze_disasters.html', data=data, magnitude=magnitude, tsunami=tsunami, flood=flood, disaster_type=disaster_type)
    else:
        # Return with empty data and flood
        return render_template('analyze_disasters.html', data=None, magnitude=[], tsunami=tsunami, flood=flood, disaster_type=disaster_type)

def generate_otp():
    return ''.join(random.choices(string.digits, k=6))

@app.route('/analyze_food_supply', methods=['GET', 'POST'])
def analyze_food_supply():
    if request.method == 'POST':
        # Get user inputs from the form
        latitude = float(request.form.get('latitude'))
        longitude = float(request.form.get('longitude'))
        disaster_type = request.form.get('disaster_type')
        severity_level = request.form.get('severity_level')
        total_population_affected = int(request.form.get('total_population_affected'))

        # Calculate alert level based on severity level
        alert_level = ''
        if severity_level == 'low':
            alert_level = 'Green'
        elif severity_level == 'medium':
            alert_level = 'Yellow'
        elif severity_level == 'high':
            alert_level = 'Orange'
        else:
            alert_level = 'Red'

        # Estimate food requirements (Assuming 2.5 kg of food per person per day)
        food_required = total_population_affected * 2.5  # in kg

        # Estimate expected delivery time based on severity level
        delivery_time = 0
        if alert_level == 'Green':
            delivery_time = 24  # 24 hours for Green
        elif alert_level == 'Yellow':
            delivery_time = 48  # 48 hours for Yellow
        elif alert_level == 'Orange':
            delivery_time = 72  # 72 hours for Orange
        else:
            delivery_time = 96  # 96 hours for Red

        # Resource allocation logic (e.g., calculate the number of trucks, personnel, etc.)
        trucks_needed = food_required / 1000  # Assuming each truck carries 1000 kg of food
        personnel_needed = total_population_affected / 100  # 1 person for every 100 affected individuals

        # Render the result
        return render_template('food_supply_result.html',
                               latitude=latitude,
                               longitude=longitude,
                               disaster_type=disaster_type.capitalize(),
                               alert_level=alert_level,
                               food_required=food_required,
                               delivery_time=delivery_time,
                               trucks_needed=trucks_needed,
                               personnel_needed=personnel_needed)

    return render_template('analyze_food_supply.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        otp = generate_otp()
        otp_expiration = datetime.utcnow() + timedelta(seconds=30)       
        # Save OTP and its expiration time
        db.collection('otps').document(email).set({
            'otp': otp,
             'expiration': otp_expiration.isoformat()
        })
        
        # Send OTP to user's email
        msg = Message('Your OTP Code', sender=app.config['MAIL_USERNAME'], recipients=[email])
        msg.body = f'Your OTP code is: {otp}. It will expire in 30 seconds.'
        mail.send(msg)
        
        session['signup_email'] = email
        return redirect(url_for('verify_otp'))

    return render_template('signup.html')

@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    if request.method == 'POST':
        email = session.get('signup_email')
        otp_entered = request.form['otp']
        password = session.get('signup_password')

        if not email:
            flash('Session expired. Please try signing up again.', 'danger')
            return redirect(url_for('signup'))

        # Retrieve stored OTP and expiration
        doc = db.collection('otps').document(email).get()
        if doc.exists:
            data = doc.to_dict()
            stored_otp = data['otp']
            expiration_str = data['expiration']

            # Convert expiration to datetime object
            expiration = datetime.fromisoformat(expiration_str)

            if otp_entered == stored_otp and datetime.utcnow() < expiration:
                # OTP is correct and not expired
                auth.create_user(email=email, password=password)
                db.collection('users').document(email).set({
                    'email': email,
                    'signup_date': datetime.utcnow().isoformat()
                })

                # Clean up OTP after successful verification
                db.collection('otps').document(email).delete()

                # Set user session
                session['user'] = email

                flash('Signup successful!', 'success')
                return redirect(url_for('main_dashboard'))
            else:
                flash('Invalid or expired OTP!', 'danger')
                return redirect(url_for('signup'))  # Redirect back to signup on failure
        else:
            flash('OTP verification failed. Please try again.', 'danger')
            return redirect(url_for('signup'))  # Redirect back to signup on failure

    return render_template('verify_otp.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        try:
            # Authenticate user with Firebase
            user = auth.get_user_by_email(email)
            # Firebase does not handle password verification directly; typically, you use Firebase Authentication SDK for login
            # For simplicity, assume login is successful if the user exists
            session['user'] = email
            return redirect(url_for('main_dashboard'))
        except auth.AuthError:
            flash('Invalid credentials, please try again.', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)