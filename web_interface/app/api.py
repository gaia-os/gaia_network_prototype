from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
import random
import math
import time

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Create sample data for demonstration purposes
def generate_sea_level_data():
    data = []
    for year in range(1990, 2051):
        value = 0.0009 * (year - 1990) * (year - 1990)
        confidence = max(0.9 - (0.005 * (year - 1990)), 0.5)
        data.append({"year": year, "value": round(value, 4), "confidence": round(confidence, 2)})
    return data

def generate_storm_data():
    data = []
    for year in range(1990, 2051):
        value = 1.0 + 0.007 * (year - 1990)
        confidence = max(0.95 - (0.006 * (year - 1990)), 0.5)
        data.append({"year": year, "value": round(value, 2), "confidence": round(confidence, 2)})
    return data

def generate_property_data():
    data = []
    for year in range(1990, 2051):
        # Initial growth followed by decline due to climate risks
        if year < 2030:
            value = 100 * (1 + 0.02 * (year - 1990))
        else:
            value = 100 * (1 + 0.02 * (2030 - 1990)) * (0.99 ** (year - 2030))
        confidence = max(0.9 - (0.005 * (year - 1990)), 0.5)
        data.append({"year": year, "value": round(value, 1), "confidence": round(confidence, 2)})
    return data

def generate_insurance_data():
    data = []
    for year in range(1990, 2051):
        # Accelerating increase in insurance premiums
        value = 1000 * (1 + 0.005 * (year - 1990) + 0.0002 * (year - 1990) ** 2)
        confidence = max(0.95 - (0.005 * (year - 1990)), 0.5)
        data.append({"year": year, "value": round(value, 1), "confidence": round(confidence, 2)})
    return data

def generate_risk_comparison():
    traditional = []
    gaia_network = []
    
    for risk_factor in range(1, 11):
        # Traditional approach has lower returns at higher risks
        trad_return = 3 + 0.2 * risk_factor
        trad_confidence = 0.9 - (0.02 * risk_factor)
        
        # Gaia Network provides better risk-adjusted returns
        gaia_return = 3 + 0.3 * risk_factor
        gaia_confidence = 0.95 - (0.015 * risk_factor)
        
        traditional.append({
            "risk_factor": risk_factor,
            "expected_return": round(trad_return, 2),
            "confidence": round(trad_confidence, 2)
        })
        
        gaia_network.append({
            "risk_factor": risk_factor,
            "expected_return": round(gaia_return, 2),
            "confidence": round(gaia_confidence, 2)
        })
    
    return {"traditional": traditional, "gaia_network": gaia_network}

def run_simulation(params):
    time_horizon = params.get("time_horizon", 30)
    adaptation_level = params.get("adaptation_level", 0.5)
    climate_scenario = params.get("climate_scenario", "RCP4.5")
    
    # Scenario factors affect simulation outcomes
    scenario_factors = {
        "RCP2.6": 0.7,
        "RCP4.5": 1.0,
        "RCP6.0": 1.3,
        "RCP8.5": 2.0
    }
    
    factor = scenario_factors.get(climate_scenario, 1.0)
    
    sea_level_data = []
    property_value_data = []
    roi_data = []
    
    current_year = 2023
    for year_offset in range(0, time_horizon + 1):
        year = current_year + year_offset
        
        # Sea level rise is mitigated by adaptation but affected by climate scenario
        sea_level = (0.01 * year_offset * factor) * (1 - adaptation_level * 0.3)
        confidence = max(0.9 - (0.01 * year_offset), 0.5)
        
        # Property value declines with sea level rise but is protected by adaptation
        property_value = 100 * (1 - (sea_level * 10) * (1 - adaptation_level * 0.5))
        
        # ROI is affected by both climate scenario and adaptation measures
        roi = 0.05 - (0.005 * year_offset * factor * (1 - adaptation_level))
        
        sea_level_data.append({
            "year": year,
            "value": round(sea_level, 3), 
            "confidence": round(confidence, 2)
        })
        
        property_value_data.append({
            "year": year,
            "value": max(round(property_value, 1), 0), 
            "confidence": round(confidence, 2)
        })
        
        roi_data.append({
            "year": year,
            "value": max(round(roi, 4), -0.1), 
            "confidence": round(confidence, 2)
        })
    
    return {
        "sea_level": sea_level_data,
        "property_value": property_value_data,
        "roi": roi_data,
        "summary": {
            "final_sea_level": sea_level_data[-1]["value"],
            "final_property_value": property_value_data[-1]["value"],
            "final_roi": roi_data[-1]["value"],
            "avg_confidence": roi_data[-1]["confidence"]
        }
    }

# API routes for node information
@app.route('/api/node-a/info', methods=['GET'])
def get_node_a_info():
    return jsonify({
        "id": "node-a",
        "name": "Real Estate Finance Node",
        "description": "Provides financial models and property valuation for coastal properties"
    })

@app.route('/api/node-b/info', methods=['GET'])
def get_node_b_info():
    return jsonify({
        "id": "node-b",
        "name": "Climate Risk Node",
        "description": "Analyzes climate change patterns and predicts effects on coastal areas"
    })

@app.route('/api/node-c/info', methods=['GET'])
def get_node_c_info():
    return jsonify({
        "id": "node-c",
        "name": "Actuarial Data Node",
        "description": "Calculates risk assessments and insurance implications of climate change"
    })

# API routes for querying nodes
@app.route('/api/node-<node_id>/query/<variable_name>', methods=['POST'])
def query_node(node_id, variable_name):
    # Simulate processing time
    time.sleep(1)
    
    # Return simulated results
    if node_id == 'a':
        return jsonify({
            "value": round(random.uniform(5.0, 8.0), 2),
            "confidence": round(random.uniform(0.7, 0.9), 2)
        })
    elif node_id == 'b':
        return jsonify({
            "value": round(random.uniform(0.5, 2.0), 2),
            "confidence": round(random.uniform(0.65, 0.85), 2)
        })
    elif node_id == 'c':
        return jsonify({
            "value": round(random.uniform(1000, 5000), 2),
            "confidence": round(random.uniform(0.75, 0.95), 2)
        })
    else:
        return jsonify({"error": "Invalid node ID"}), 400

# Historical data endpoints
@app.route('/api/historical/sea_level_rise', methods=['GET'])
def historical_sea_level():
    return jsonify(generate_sea_level_data())

@app.route('/api/historical/storm_intensity', methods=['GET'])
def historical_storm():
    return jsonify(generate_storm_data())

@app.route('/api/historical/property_values', methods=['GET'])
def historical_property():
    return jsonify(generate_property_data())

@app.route('/api/historical/insurance_premiums', methods=['GET'])
def historical_insurance():
    return jsonify(generate_insurance_data())

# Risk comparison endpoint
@app.route('/api/risk-comparison', methods=['GET'])
def risk_comparison():
    return jsonify(generate_risk_comparison())

# Update CORS configuration to explicitly handle OPTIONS
CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"]}})

# Add a health check endpoint
@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health_check():
    if request.method == 'OPTIONS':
        return '', 200
    return jsonify({"status": "ok"})

# Update simulation endpoint to use the expected path
@app.route('/api/run_simulation', methods=['POST', 'OPTIONS'])
def run_simulation_api():
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        return '', 200
    
    # Get request data
    params = request.json
    
    # Check if this is just a connectivity test
    if params and 'check_only' in params and params['check_only']:
        return jsonify({"status": "ok"})
    
    # Use the existing simulation function
    return jsonify(run_simulation(params))

# Simulation endpoint (keeping for backward compatibility)
@app.route('/api/simulation', methods=['POST'])
def simulation():
    params = request.json
    return jsonify(run_simulation(params))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3030, debug=False) 