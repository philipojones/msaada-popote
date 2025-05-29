from flask import Flask, request
import africastalking
import os
from dotenv import load_dotenv
import pandas as pd
import pickle as pk
import logging
from datetime import datetime
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hospital_ussd.log"),
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
ussd_sessions = {}

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
            "distance_km": distance_km
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {"success": False, "error": str(e)}

@app.route('/ussd', methods=['POST', 'GET'])
def hospital_ussd():
    """USSD callback endpoint for hospital location service"""
    session_id = request.values.get("sessionId", "")
    service_code = request.values.get("serviceCode", "*123#")
    phone_number = request.values.get("phoneNumber", "")
    text = request.values.get("text", "").strip()
    
    # Initialize or retrieve session
    if session_id not in ussd_sessions:
        ussd_sessions[session_id] = {
            "phone_number": phone_number,
            "current_step": "welcome",
            "data": {}
        }
    
    session = ussd_sessions[session_id]
    response = ""
    
    # USSD menu flow
    if text == "":
        session["current_step"] = "welcome"
        response = "CON Karibu kwenye Huduma ya Hospitali:\n"
        response += "1. Tafuta Hospitali ya Karibu\n"
        response += "2. Maelezo ya Huduma\n"
        response += "3. Msaada"
    
    elif text == "1":
        session["current_step"] = "select_region"
        response = "CON Chagua Mkoa:\n"
        response += "1. Dar es Salaam\n"
        response += "2. Arusha\n"
        response += "3. Mwanza"
    
    elif text == "1*1" or text == "1*2" or text == "1*3":
        region_map = {"1*1": "Dar es Salaam", "1*2": "Arusha", "1*3": "Mwanza"}
        session["data"]["region"] = region_map[text]
        session["current_step"] = "select_district"
        districts = data[data['Region'] == session["data"]["region"]]['District'].unique()
        response = "CON Chagua Wilaya:\n"
        for i, district in enumerate(districts, 1):
            response += f"{i}. {district}\n"
    
    elif text.startswith("1*") and len(text.split('*')) == 3:
        try:
            region_idx = text.split('*')[1]
            district_idx = int(text.split('*')[2]) - 1
            region_map = {"1": "Dar es Salaam", "2": "Arusha", "3": "Mwanza"}
            region = region_map.get(region_idx)
            districts = data[data['Region'] == region]['District'].unique()
            if 0 <= district_idx < len(districts):
                session["data"]["district"] = districts[district_idx]
                session["current_step"] = "select_street"
                streets = data[data['District'] == session["data"]["district"]]['Street'].unique()
                response = "CON Chagua Mtaa:\n"
                for i, street in enumerate(streets, 1):
                    response += f"{i}. {street}\n"
            else:
                response = "END Chaguo si sahihi. Tafadhali anza tena."
        except:
            response = "END Chaguo si sahihi. Tafadhali anza tena."
    
    elif text.startswith("1*") and len(text.split('*')) == 4:
        try:
            region_idx = text.split('*')[1]
            district_idx = int(text.split('*')[2]) - 1
            street_idx = int(text.split('*')[3]) - 1
            region_map = {"1": "Dar es Salaam", "2": "Arusha", "3": "Mwanza"}
            region = region_map.get(region_idx)
            districts = data[data['Region'] == region]['District'].unique()
            if 0 <= district_idx < len(districts):
                district = districts[district_idx]
                streets = data[data['District'] == district]['Street'].unique()
                if 0 <= street_idx < len(streets):
                    session["data"]["street"] = streets[street_idx]
                    prediction = predict_nearest_hospital(
                        session["data"]["street"],
                        session["data"]["district"],
                        session["data"]["region"]
                    )
                    if prediction["success"]:
                        sms_message = (
                            f"Hospitali ya Karibu kutoka:\n"
                            f"Mtaa Wako: {session['data']['street']}\n"
                            f"Wilaya Yako: {session['data']['district']}\n"
                            f"Mkoa Wako: {session['data']['region']}\n\n"
                            f"Jina la Hospitali: {prediction['hospital_name']}\n"
                            f"Mtaa: {prediction['street']}\n"
                            f"Wilaya: {prediction['district']}\n"
                            f"Mkoa: {prediction['region']}\n"
                            f"Umbali (takriban): {prediction['distance_km']:.2f} km\n"
                            f"Asante kwa kutumia huduma yetu!"
                        )
                        sms_result = send_sms(phone_number, sms_message)
                        if sms_result["success"]:
                            response = (
                                f"END Hospitali ya karibu: {prediction['hospital_name']}\n"
                                f"SMS imetumwa kwa {phone_number}"
                            )
                        else:
                            response = (
                                f"END Hospitali ya karibu: {prediction['hospital_name']}\n"
                                f"Hitilafu: SMS haijatumwa. Jaribu tena."
                            )
                            logger.error(f"SMS failed for {phone_number}: {sms_result['error']}")
                        
                        # Clean up session
                        del ussd_sessions[session_id]
                    else:
                        response = f"END Samahani: {prediction['error']}"
                else:
                    response = "END Chaguo si sahihi. Tafadhali anza tena."
            else:
                response = "END Chaguo si sahihi. Tafadhali anza tena."
        except Exception as e:
            logger.error(f"USSD prediction failed: {str(e)}")
            response = "END Samahani, kuna tatizo la kiufundi. Jaribu tena."
    
    elif text == "2":
        response = "END Huduma yetu inakusaidia kupata hospitali ya karibu kwa haraka. Tuma *123# kuanza."
    
    elif text == "3":
        response = "END Msaada:\n1. Chagua 'Tafuta Hospitali'\n2. Fuata maagizo ya mkoa, wilaya, na mtaa\n3. Pokea maelezo kwenye SMS\n*123# kuanza"
    
    else:
        response = "END Mchujo si sahihi. Tuma *123# kuanza tena."
    
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))