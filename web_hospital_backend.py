from flask import Flask, request, jsonify, render_template_string
import africastalking
import os
import re
from dotenv import load_dotenv
import pandas as pd
import pickle as pk
import logging
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hospital_web.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Africa's Talking
ussdusername = "sandbox"
ussd_key = os.getenv("AT_API_KEY")
africastalking.initialize(ussdusername, ussd_key)
sms = africastalking.SMS

# Load the trained model, encoders, and dataset
try:
    model = pk.load(open('hospital_prediction_model.pkl', 'rb'))
    le_street = pk.load(open('le_street.pkl', 'rb'))
    le_district = pk.load(open('le_district.pkl', 'rb'))
    le_region = pk.load(open('le_region.pkl', 'rb'))
    data = pk.load(open('hospital_data.pkl', 'rb'))
    logger.info("Model, encoders, and dataset loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model, encoders, or dataset: {str(e)}")
    raise RuntimeError("Failed to load prediction components")

app = Flask(__name__)

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

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in kilometers using haversine formula"""
    R = 6371  # Earth's radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def send_sms(phone_number: str, message: str) -> dict:
    """Send SMS via Africa's Talking with enhanced error handling"""
    try:
        if not phone_number.startswith('+'):
            phone_number = f"+255{phone_number.lstrip('0')}"
        response = sms.send(message, [phone_number])
        logger.info(f"SMS sent to {phone_number}: {response}")
        return {
            "success": True,
            "response": response,
            "recipient": phone_number
        }
    except Exception as e:
        logger.error(f"SMS sending failed to {phone_number}: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "recipient": phone_number
        }

def predict_nearest_hospital(street: str, district: str, region: str) -> dict:
    """Predict the nearest hospital using the trained model"""
    try:
        if street not in street_offsets or (district, region) not in coordinate_map:
            return {"success": False, "error": "Invalid street, district, or region provided"}
        
        base_coords = coordinate_map[(district, region)]
        street_offset = street_offsets.get(street, (0, 0))
        input_lat = base_coords[0] + street_offset[0]
        input_lon = base_coords[1] + street_offset[1]
        input_coords = np.array([[input_lat, input_lon]])
        
        distances, indices = model.kneighbors(input_coords)
        nearest_hospital = data.iloc[indices[0][0]]
        
        # Calculate distance in kilometers
        hospital_lat = nearest_hospital['Latitude']
        hospital_lon = nearest_hospital['Longitude']
        distance_km = haversine_distance(input_lat, input_lon, hospital_lat, hospital_lon)
        
        return {
            "success": True,
            "hospital_name": nearest_hospital['Hospital Name'],
            "street": nearest_hospital['Street'],
            "district": nearest_hospital['District'],
            "region": nearest_hospital['Region'],
            "latitude": hospital_lat,
            "longitude": hospital_lon,
            "distance": distance_km
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {"success": False, "error": str(e)}

@app.route('/')
def index():
    """Render the web interface"""
    with open('index.html', 'r') as f:
        template = f.read()
    return render_template_string(template)

@app.route('/chatbot')
def chatbot():
    """Render the chatbot interface"""
    with open('chatbot.html', 'r') as f:
        template = f.read()
    return render_template_string(template)

@app.route('/find_hospital', methods=['POST'])
def find_hospital():
    """Handle web form submission and send SMS"""
    try:
        data = request.get_json()
        region = data['region']
        district = data['district']
        street = data['street']
        phone = data['phone']
        
        prediction = predict_nearest_hospital(street, district, region)
        if prediction['success']:
            sms_message = (
                f"Hospitali ya Karibu kutoka:\n"
                f"Mtaa Wako: {street}\n"
                f"Wilaya Yako: {district}\n"
                f"Mkoa Wako: {region}\n\n"
                f"Jina la Hospitali: {prediction['hospital_name']}\n"
                f"Mtaa: {prediction['street']}\n"
                f"Wilaya: {prediction['district']}\n"
                f"Mkoa: {prediction['region']}\n"
                f"Umbali (takriban): {prediction['distance']:.2f} km\n"
                f"Asante kwa kutumia huduma yetu!"
            )
            sms_result = send_sms(phone, sms_message)
            if sms_result['success']:
                return jsonify({
                    "success": True,
                    "hospital_name": prediction['hospital_name'],
                    "street": prediction['street'],
                    "district": prediction['district'],
                    "region": prediction['region'],
                    "distance": prediction['distance']
                })
            else:
                return jsonify({"success": False, "error": sms_result['error']})
        else:
            return jsonify({"success": False, "error": prediction['error']})
    except Exception as e:
        logger.error(f"Web prediction failed: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/send_sms', methods=['POST'])
def send_sms_endpoint():
    """Handle SMS sending via POST request"""
    try:
        data = request.get_json()
        phone_number = data.get('phone_number')
        message = data.get('message')

        # Validate inputs
        if not phone_number or not message:
            return jsonify({"success": False, "error": "Phone number and message are required"}), 400
        
        # Validate phone number format (Tanzanian numbers: +255 followed by 9 digits)
        phone_pattern = re.compile(r'^\+255[67]\d{8}$')
        if not phone_pattern.match(phone_number):
            return jsonify({"success": False, "error": "Invalid phone number format. Use +255 followed by 9 digits (e.g., +255712345678)"}), 400
        
        # Validate message length (Africa's Talking SMS limit is typically 160 characters for single SMS)
        if len(message) > 160:
            return jsonify({"success": False, "error": "Message exceeds 160 character limit"}), 400

        sms_result = send_sms(phone_number, message)
        if sms_result['success']:
            return jsonify({"success": True, "recipient": sms_result['recipient'], "message": "SMS sent successfully"}), 200
        else:
            return jsonify({"success": False, "error": sms_result['error'], "recipient": sms_result['recipient']}), 500
    except Exception as e:
        logger.error(f"SMS endpoint failed: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))