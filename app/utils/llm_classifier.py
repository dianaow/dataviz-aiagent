import os
import pandas as pd
import json
import google.generativeai as genai
from dotenv import load_dotenv
from dash import html, dcc
import dash_bootstrap_components as dbc
import json
from jsonschema import validate, ValidationError

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Gemini model
model = genai.GenerativeModel('gemini-2.0-flash')

# Define the supported data types
SUPPORTED_DATA_TYPES = [
    "Battery Cycling Data",
    "X-ray Diffraction",
    "Raman Spectroscopy"
]

# Define expected column patterns for each data type
COLUMN_PATTERNS = {
    "Battery Cycling Data": {
        "required": [
            ["cycle", "index"],
            ["current", "i"],
            ["voltage", "v"],
            ["capacity", "cap"],
            ["time", "date"]
        ],
        "optional": [
            ["energy", "power"],
            ["resistance", "impedance"],
            ["temperature", "temp"]
        ]
    },
    "X-ray Diffraction": {
        "required": [
            ["2theta", "theta", "angle"],
            ["intensity", "counts", "signal"]
        ],
        "optional": [
            ["phase", "composition"],
            ["temperature", "temp"]
        ]
    },
    "Raman Spectroscopy": {
        "required": [
            ["wavenumber", "wave", "raman", "shift"],
            ["intensity", "counts", "signal"]
        ],
        "optional": [
            ["phase", "composition"],
            ["temperature", "temp"]
        ]
    }
}

def get_column_analysis(df):
    """Generate a summary of the DataFrame columns for LLM analysis."""
    column_info = []
    
    for col in df.columns:
        # Get data type
        dtype = str(df[col].dtype)
        
        # Get min, max, mean, etc. for numeric columns
        stats = {}
        if pd.api.types.is_numeric_dtype(df[col]):
            stats = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "unique_values": int(df[col].nunique())
            }
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            # Handle datetime columns
            stats = {
                "min": df[col].min().strftime('%Y-%m-%d %H:%M:%S'),
                "max": df[col].max().strftime('%Y-%m-%d %H:%M:%S'),
                "unique_values": int(df[col].nunique()),
                "sample_values": df[col].dropna().head(3).dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
            }
        else:
            # For non-numeric columns, just count unique values
            stats = {
                "unique_values": int(df[col].nunique()),
                "sample_values": df[col].dropna().head(3).astype(str).tolist()
            }
        
        column_info.append({
            "name": col,
            "dtype": dtype,
            "stats": stats
        })
    
    return column_info

def analyze_column_matches(df, data_type):
    """Analyze how well the columns match expected patterns for a data type using LLM."""
    matches = {
        "required": [],
        "optional": [],
        "missing": [],
        "confidence": 0.0
    }
    
    if data_type not in COLUMN_PATTERNS:
        return matches
    
    try:
        # Create a summary of the columns for LLM analysis
        column_summary = []
        for col in df.columns:
            # Get data type and sample values
            dtype = str(df[col].dtype)
            # Convert timestamp values to strings if present
            sample_values = df[col].dropna().head(3).apply(
                lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.api.types.is_datetime64_any_dtype(df[col]) else x
            ).tolist()
            
            column_summary.append({
                "name": col,
                "type": dtype,
                "sample_values": sample_values
            })
        
        # Prepare the prompt for Gemini
        prompt = f"""You are an expert in data analysis and column mapping.
        Your task is to analyze the provided columns and match them with the expected patterns for {data_type}.
        For each column, determine if it matches any of the required or optional patterns.

        Column information: {json.dumps(column_summary, indent=2)}

        Expected patterns for {data_type}:
        Required: {COLUMN_PATTERNS[data_type]['required']}
        Optional: {COLUMN_PATTERNS[data_type]['optional']}

        Respond with a JSON object containing:
        - required: list of matched required columns with confidence scores
        - optional: list of matched optional columns with confidence scores
        - missing: list of required patterns that weren't matched
        - confidence: overall confidence score (0-1)

        Format each match as: {{"old_column": "original_name", "new_column": "matched_name", "confidence": 0.X}}
        """
        
        # Call Gemini API
        response = model.generate_content(prompt)
        
        # Parse the response
        result = json.loads(response.text)
        
        # Validate and structure the matches
        matches["required"] = result.get("required", [])
        matches["optional"] = result.get("optional", [])
        matches["missing"] = result.get("missing", [])
        matches["confidence"] = result.get("confidence", 0.0)
        return matches
        
    except Exception as e:
        print(f"Error in LLM column analysis: {e}")
        # Fallback to rule-based matching
        return rule_based_column_matching(df, data_type)

def classify_data(df, user_hint=None, filename=None):
    """
    Use LLM to classify the type of data in the DataFrame.
    Returns: 
        tuple: (data_type, confidence, column_matches, needs_confirmation)
    """
    # If user provided a hint, we prioritize that
    if user_hint:
        for data_type in SUPPORTED_DATA_TYPES:
            if data_type.lower() in user_hint.lower():
                matches = analyze_column_matches(df, data_type)
                return data_type, 0.95, matches, False
    
    # Check if API key is available
    if not os.getenv("GOOGLE_API_KEY"):
        # Fallback to rule-based classification if no API key
        data_type = rule_based_classification(df, filename)
        matches = analyze_column_matches(df, data_type)
        return data_type, 0.7, matches, True
    
    try:
        # Generate column analysis for the LLM
        column_analysis = get_column_analysis(df)
        
        # Create a sample of the data, converting timestamps to strings
        data_sample = df.copy()
        for col in data_sample.columns:
            if pd.api.types.is_datetime64_any_dtype(data_sample[col]):
                data_sample[col] = data_sample[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        data_sample = data_sample.head(5).to_dict(orient='records')
        
        # Prepare the prompt for Gemini
        prompt = f"""You are an expert in materials science and metrology data. 
        Your task is to classify the type of experimental data provided based on the column names, data patterns, and sample values.
        
        File name: {filename if filename else 'Not provided'}
        Column information: {json.dumps(column_analysis, indent=2)}
        Data sample: {json.dumps(data_sample, indent=2)}
        
        The supported data types are: {', '.join(SUPPORTED_DATA_TYPES)}.
        If you cannot determine the type with confidence, classify it as 'Other'.
        
        Respond with ONLY the data type and a confidence level (0-1) in JSON format: {{"data_type": "TYPE", "confidence": 0.X}}"""
        
        # Call Gemini API
        response = model.generate_content(prompt)
        
        try:
            # Parse the JSON response
            result = json.loads(response.text)
            data_type = result.get("data_type", "Other")
            confidence = result.get("confidence", 0.5)
            
            # Validate the data type
            if data_type not in SUPPORTED_DATA_TYPES:
                data_type = "Other"
                confidence = 0.5
            
            # Analyze column matches
            matches = analyze_column_matches(df, data_type)
            
            # Determine if we need user confirmation
            needs_confirmation = (
                confidence < 0.8 or  # Low confidence in classification
                len(matches["missing"]) > 0 or  # Missing required columns
                len(matches["required"]) < 2  # Not enough required columns found
            )
            
            return data_type, confidence, matches, needs_confirmation
            
        except json.JSONDecodeError:
            # Fallback to rule-based classification if JSON parsing fails
            data_type = rule_based_classification(df, filename)
            matches = analyze_column_matches(df, data_type)
            return data_type, 0.6, matches, True
            
    except Exception as e:
        print(f"Error in LLM classification: {e}")
        # Fallback to rule-based classification
        data_type = rule_based_classification(df, filename)
        matches = analyze_column_matches(df, data_type)
        return data_type, 0.5, matches, True

def create_column_confirmation_dialog(matches, data_type, chat_history=None):
    """Create a chatbot-style dialog component for column confirmation."""
    if chat_history is None:
        chat_history = []
    print(f"Matches: {matches}")
    # Create a container for the chat messages
    chat_container = html.Div([
        # Confirmation button container
        html.Div([
            dbc.Button("Confirm", color="success", id="confirm-classification", size="md")
        ], className="d-flex justify-content-center"),

        # Chat messages container with scrolling
        html.Div([
            # Display chat history
            html.Div([
                html.Div([
                    html.Div([
                        html.P(message["content"], className="mb-0"),
                        html.Div([
                            "Here is the data type I found: ",
                            html.Strong(data_type)
                        ], className="mb-2"),
                        html.P("Here are the column matches I found:", className="mb-2"),
                        html.Div([
                            html.Strong("Required columns:", className="d-block mb-2"),
                            html.Ul([
                                html.Li(f"{match.get('old_column', 'Unknown')}: {match.get('new_column', 'Unknown')} "
                                      f"(Confidence: {match.get('confidence', 0):.0%})")
                                for match in matches.get("required", [])
                            ], className="mb-3"),
                            html.Strong("Optional columns:", className="d-block mb-2") if matches.get("optional") else None,
                            html.Ul([
                                html.Li(f"{match.get('old_column', 'Unknown')}: {match.get('new_column', 'Unknown')} "
                                      f"(Confidence: {match.get('confidence', 0):.0%})")
                                for match in matches.get("optional", [])
                            ], className="mb-3"),
                            html.Strong("Missing required columns:", className="d-block mb-2") if matches.get("missing") else None,
                            html.Ul([
                                html.Li(missing) for missing in matches.get("missing", [])
                            ]) if matches.get("missing") else None
                        ], className="column-matches-content") if matches else None,
                        html.Small(message.get("timestamp", ""), className="text-muted d-block mt-1")
                    ], className="chat-bubble assistant-bubble p-3 rounded-3")
                ], className="chat-message assistant-message mb-3") if message["role"] == "assistant"
                else html.Div([
                    html.Div([
                        html.P(message["content"], className="mb-0"),
                        html.Small(message.get("timestamp", ""), className="text-muted d-block mt-1")
                    ], className="chat-bubble user-bubble p-3 rounded-3")
                ], className="chat-message user-message mb-3")
                for message in chat_history
            ], className="chat-messages-container"),
        ], className="chat-messages-wrapper p-3"),
        
        # Input area with send button
        html.Div([
            html.Div([
                dbc.Input(
                    id="user-feedback-input",
                    type="text",
                    placeholder="Type your feedback here",
                    className="chat-input"
                ),
                dbc.Button("Send", id="send-feedback-button", color="primary", size="sm", className="ms-2")
            ], className="chat-input-container d-flex align-items-center")
        ], className="chat-input-wrapper p-3 border-top"),
      
    ], className="chat-container border rounded-3")
    
    return chat_container

def process_user_feedback(feedback, current_matches, data_type):
    """Process user feedback and update column matches."""
    try:
        # Create a summary of the current state
        current_state = {
            "data_type": data_type,
            "current_matches": current_matches,
            "user_feedback": feedback
        }
        
        # Prepare the prompt for Gemini
        prompt = f"""Your task is to process user feedback about data classification, column matches and suggest corrections.
        If the user indicates a specific data type or column should be used, update the matches accordingly.
        
        Current state: {json.dumps(current_state, indent=2)}
        
        Respond with a JSON object containing:
        - updated_data_type: updated data type
        - updated_matches: list of updated column matches
        - explanation: explanation of the changes made
        - needs_confirmation: boolean indicating if further confirmation is needed"""
        
        # Call Gemini API
        response = model.generate_content(prompt)
        
        # Parse the response
        result = json.loads(response.text)
        
        return {
            "updated_data_type": result.get("updated_data_type", data_type),
            "updated_matches": result.get("updated_matches", current_matches),
            "explanation": result.get("explanation", ""),
            "needs_confirmation": result.get("needs_confirmation", True)
        }
        
    except Exception as e:
        print(f"Error processing user feedback: {e}")
        return {
            "updated_matches": current_matches,
            "explanation": "Sorry, I couldn't process your feedback. Please try again.",
            "needs_confirmation": True
        }

def rule_based_classification(df, filename=None):
    """Simple rule-based fallback classification when LLM is not available."""
    
    # Check filename for clues
    if filename:
        filename_lower = filename.lower()
        if any(term in filename_lower for term in ['battery', 'cycle', 'cycling', 'charge', 'discharge']):
            return "Battery Cycling Data"
        elif any(term in filename_lower for term in ['xrd', 'diffraction', 'xray']):
            return "X-ray Diffraction"
        elif any(term in filename_lower for term in ['raman', 'spectroscopy']):
            return "Raman Spectroscopy"
    
    # Check column names for clues
    column_names = ' '.join(df.columns).lower()
    
    if any(term in column_names for term in ['cycle', 'capacity', 'voltage', 'current', 'charge', 'discharge']):
        return "Battery Cycling Data"
    elif any(term in column_names for term in ['2theta', 'angle', 'diffraction', 'xrd']):
        return "X-ray Diffraction"
    elif any(term in column_names for term in ['raman', 'shift', 'wavenumber']):
        return "Raman Spectroscopy"
    
    # If no patterns detected, default to "Other"
    return "Other"

def rule_based_column_matching(df, data_type):
    """Fallback rule-based column matching when LLM is not available."""
    matches = {
        "required": [],
        "optional": [],
        "missing": [],
        "confidence": 0.0
    }
    
    if data_type not in COLUMN_PATTERNS:
        return matches
    
    # Check required columns
    for patterns in COLUMN_PATTERNS[data_type]["required"]:
        found = False
        for pattern in patterns:
            for col in df.columns:
                if pattern.lower() in col.lower():
                    matches["required"].append({
                        "new_column": pattern,
                        "old_column": col,
                        "confidence": 1.0
                    })
                    found = True
                    break
            if found:
                break
        if not found:
            matches["missing"].append(patterns[0])
    
    # Check optional columns
    for patterns in COLUMN_PATTERNS[data_type]["optional"]:
        for pattern in patterns:
            for col in df.columns:
                if pattern.lower() in col.lower():
                    matches["optional"].append({
                        "new_column": pattern,
                        "old_column": col,
                        "confidence": 0.8
                    })
                    break
    
    # Calculate overall confidence
    total_required = len(COLUMN_PATTERNS[data_type]["required"])
    found_required = len(matches["required"])
    if total_required > 0:
        matches["confidence"] = found_required / total_required
    
    return matches