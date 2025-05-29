from flask import Flask, request, jsonify
import africastalking
import os
from dotenv import load_dotenv
import pandas as pd
import pickle as pk
import logging
from datetime import datetime, timedelta
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hospital_sms_chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
app = Flask(__name__)

# Initialize Africa's Talking with production credentials
USERNAME = "sandbox" # Your production username
API_KEY = os.getenv("AT_API_KEY")    # Your production API key
SHORTCODE = os.getenv("AT_SHORTCODE")  # Your shortcode (e.g., "15086")

africastalking.initialize(USERNAME, API_KEY)
sms = africastalking.SMS

# Load model and encoders
try:
    model = pk.load(open('hospital_prediction_model.pkl', 'rb'))
    le_street = pk.load(open('le_street.pkl', 'rb'))
    le_district = pk.load(open('le_district.pkl', 'rb'))
    le_region = pk.load(open('le_region.pkl', 'rb'))
    data = pk.load(open('hospital_data.pkl', 'rb'))
    logger.info("Model and data loaded successfully")
except Exception as e:
    logger.error(f"Failed to load components: {str(e)}")
    raise RuntimeError("Loading failed")

# Coordinate mapping
coordinate_map = {
    ('Temeke', 'Dar es Salaam'): (-6.85, 39.35),
    ('Kinondoni', 'Dar es Salaam'): (-6.79, 39.25),
    ('Ilala', 'Dar es Salaam'): (-6.82, 39.20),
    ('Arusha Urban', 'Arusha'): (-3.37, 36.68),
    ('Arumeru', 'Arusha'): (-3.30, 36.85),
    ('Longido', 'Arusha'): (-2.73, 36.70),
    ('Nyamagana', 'Mwanza'): (-2.52, 32.90),
    ('Ilemela', 'Mwanza'): (-2.48, 32.95),
    ('Magu', 'Mwanza'): (-2.60, 33.10)
}

street_offsets = {
    'Mtaa wa Amani': (0.01, 0.01),
    'Jamhuri Avenue': (0.02, -0.01),
    'Bismarck Street': (-0.01, 0.02),
    'Nyerere Street': (0.015, -0.015),
    'Nyamagana Road': (-0.02, 0.01),
    'Buhongwa Road': (0.01, -0.02),
    'Kaloleni Street': (-0.015, 0.015),
    'Njiro Road': (0.02, 0.02),
    'Meru Road': (-0.01, -0.01),
    'Mlimani Avenue': (0.015, 0.015),
    'Azikiwe Street': (-0.02, -0.02),
    'Kijitonyama Road': (0.01, 0.02),
    'Mtaa wa Upendo': (-0.015, -0.015),
    'Mtaa wa Baraka': (0.02, -0.02),
    'Mtaa wa Walinzi': (-0.01, 0.01)
}

# Session management
sms_sessions = {}
SESSION_TIMEOUT = timedelta(minutes=10)

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def send_sms(phone, message):
    """Send SMS via Africa's Talking shortcode"""
    try:
        # Format phone number
        if not phone.startswith('+'):
            phone = f"+255{phone.lstrip('0')}"
        
        # Send message with proper arguments
        response = sms.send(message, [phone], SHORTCODE)
        
        logger.info(f"SMS sent to {phone} via {SHORTCODE}. Response: {response}")
        return {"success": True, "response": response}
    except Exception as e:
        logger.error(f"SMS sending failed to {phone}: {str(e)}")
        return {"success": False, "error": str(e)}

def predict_nearest_hospital(street, district, region):
    try:
        if street not in street_offsets or (district, region) not in coordinate_map:
            return {"success": False, "error": "Mtaa, wilaya, au mkoa sio sahihi"}

        base_coords = coordinate_map[(district, region)]
        lat, lon = base_coords[0] + street_offsets[street][0], base_coords[1] + street_offsets[street][1]
        input_coords = np.array([[lat, lon]])
        distances, indices = model.kneighbors(input_coords)
        nearest = data.iloc[indices[0][0]]
        distance_km = haversine_distance(lat, lon, nearest['Latitude'], nearest['Longitude'])

        return {
            "success": True,
            "hospital_name": nearest['Hospital Name'],
            "street": nearest['Street'],
            "district": nearest['District'],
            "region": nearest['Region'],
            "latitude": nearest['Latitude'],
            "longitude": nearest['Longitude'],
            "distance_km": distance_km
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {"success": False, "error": str(e)}

def clean_expired_sessions():
    now = datetime.now()
    expired = [phone for phone, session in sms_sessions.items() if now - session["timestamp"] > SESSION_TIMEOUT]
    for phone in expired:
        del sms_sessions[phone]
        logger.info(f"Expired session cleared for {phone}")

@app.route('/sms', methods=['POST'])
def sms_chatbot():
    # Handle both sandbox and production formats
    sender = request.form.get("from") or request.form.get("phoneNumber")
    message = request.form.get("text") or request.form.get("message")

    if not sender or not message:
        send_sms(sender, "Ujumbe sio sahihi. Tuma 'hi' kuanza.")
        return jsonify({"status": "error"}), 400

    clean_expired_sessions()
    message = message.strip().lower()
    now = datetime.now()

    if sender not in sms_sessions or message == "hi":
        sms_sessions[sender] = {"step": "select_region", "data": {}, "timestamp": now}
        regions = sorted(data['Region'].unique())
        msg = "Karibu! Chagua mkoa wako:\n"
        msg += '\n'.join([f"{i+1}. {r}" for i, r in enumerate(regions)])
        msg += "\nTuma nambari ya mkoa (mfano: 1)"
        send_sms(sender, msg)
        return jsonify({"status": "ok"}), 200

    session = sms_sessions[sender]
    session["timestamp"] = now

    if session["step"] == "select_region":
        try:
            idx = int(message) - 1
            regions = sorted(data['Region'].unique())
            if 0 <= idx < len(regions):
                session["data"]["region"] = regions[idx]
                session["step"] = "select_district"
                districts = sorted(data[data['Region'] == regions[idx]]['District'].unique())
                msg = f"Umechagua: {regions[idx]}\nChagua wilaya:\n"
                msg += '\n'.join([f"{i+1}. {d}" for i, d in enumerate(districts)])
                msg += "\nTuma nambari ya wilaya"
                send_sms(sender, msg)
            else:
                send_sms(sender, "Nambari ya mkoa sio sahihi. Tuma 'hi' kuanza tena.")
        except ValueError:
            send_sms(sender, "Tuma nambari ya mkoa (mfano: 1)")
        return jsonify({"status": "ok"}), 200

    if session["step"] == "select_district":
        try:
            idx = int(message) - 1
            region = session["data"]["region"]
            districts = sorted(data[data['Region'] == region]['District'].unique())
            if 0 <= idx < len(districts):
                session["data"]["district"] = districts[idx]
                session["step"] = "enter_street"
                send_sms(sender, f"Umechagua: {districts[idx]}\nTuma jina la mtaa wako:")
            else:
                send_sms(sender, "Nambari ya wilaya sio sahihi. Tuma 'hi' kuanza tena.")
        except ValueError:
            send_sms(sender, "Tuma nambari ya wilaya (mfano: 1)")
        return jsonify({"status": "ok"}), 200

    if session["step"] == "enter_street":
        street = message.strip().title()
        region = session["data"]["region"]
        district = session["data"]["district"]
        if street in data[data['District'] == district]['Street'].unique():
            session["data"]["street"] = street
            prediction = predict_nearest_hospital(street, district, region)
            if prediction["success"]:
                reply = (
                    f"🏥 Hospitali ya karibu:\n"
                    f"{prediction['hospital_name']}\n"
                    f"{prediction['street']}, {prediction['district']}, {prediction['region']}\n"
                    f"Umbali: {prediction['distance_km']:.2f} km"
                )
            else:
                reply = prediction.get("error", "Hitilafu imetokea.")
            send_sms(sender, reply)
            del sms_sessions[sender]
        else:
            send_sms(sender, "Mtaa haupatikani. Tafadhali andika jina sahihi au tuma 'hi' kuanza tena.")
        return jsonify({"status": "ok"}), 200

    send_sms(sender, "Hitilafu. Tuma 'hi' kuanza upya.")
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    # Run in production mode
    app.run(debug=False, host='0.0.0.0', port=5000)