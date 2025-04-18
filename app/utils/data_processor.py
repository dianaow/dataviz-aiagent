import base64
import io
import pandas as pd
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Gemini model
model = genai.GenerativeModel('gemini-2.0-flash')

def parse_contents(contents, filename):
    """Parse the uploaded file contents."""
    print(f"Parsing file: {filename}")
    print(f"Contents type: {type(contents)}")
    
    if contents is None:
        print("No contents provided")
        return None
    
    content_type, content_string = contents.split(',')
    print(f"Content type: {content_type}")
    
    decoded = base64.b64decode(content_string)
    print("Decoded base64 content")
    
    try:
        if 'csv' in filename:
            print("Processing CSV file")
            # First, get the raw content as text
            raw_content = decoded.decode('utf-8')
            
            # Split into lines for processing
            lines = raw_content.split('\n')
            
            # Create a prompt for the LLM to analyze the raw content
            prompt = f"""
            Analyze this raw CSV content and:
            1. Identify which lines are metadata (should be removed)
            2. Identify which line contains the actual data headers
            3. Suggest proper column names if missing
            
            Content:
            {raw_content}
            
            IMPORTANT: You must respond with a valid JSON object in the following format:
            {{
                "metadata_lines": [list of line numbers to remove (0-based)],
                "header_line": number (line number containing headers, -1 if no headers),
                "column_names": [list of suggested column names if headers are missing],
                "data_start_line": number (line number where actual data begins)
            }}
            
            Do not include any other text in your response. Only return the JSON object.
            """
            
            # Get LLM response
            response = model.generate_content(prompt)
            print(f"LLM response: {response.text}")
            
            # Clean the response text
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]  # Remove ```json
            if response_text.startswith('```'):
                response_text = response_text[3:]  # Remove ```
            if response_text.endswith('```'):
                response_text = response_text[:-3]  # Remove ```
            response_text = response_text.strip()
            
            # Parse the response
            result = json.loads(response_text)
            
            # Process the content based on LLM analysis
            cleaned_lines = []
            for i, line in enumerate(lines):
                if i not in result["metadata_lines"]:
                    if i == result["header_line"] or (result["header_line"] == -1 and i == result["data_start_line"]):
                        # This is either the header line or the first data line
                        if result["header_line"] == -1:
                            # No headers, use suggested column names
                            cleaned_lines.append(','.join(result["column_names"]))
                        cleaned_lines.append(line)
                    elif i >= result["data_start_line"]:
                        # This is a data line
                        cleaned_lines.append(line)
            
            # Join the cleaned lines back together
            cleaned_content = '\n'.join(cleaned_lines)
            
            # Now read the cleaned content as CSV
            df = pd.read_csv(io.StringIO(cleaned_content))
            
            print('Final result', df.head())
            
        elif 'xls' in filename or 'xlsx' in filename:
            print("Processing Excel file")
            # First, read the Excel file
            excel_file = io.BytesIO(decoded)
            xls = pd.ExcelFile(excel_file)
            
            # First, identify which tabs contain data
            data_tabs = {}
            for sheet_name in xls.sheet_names:
                print(f"Analyzing sheet: {sheet_name}")
                
                # Read the first few rows of each sheet
                df = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=5)
                
                # Create a prompt for the LLM to analyze if this is a data tab
                prompt = f"""
                Analyze this Excel sheet and determine if it contains actual data or is just metadata.
                
                Sheet Name: {sheet_name}
                Content:
                {df.head().to_string()}
                
                A data tab should have:
                1. Column headers in the first row
                2. Data values starting from the second row
                3. Consistent data types in each column
                4. No large blocks of text or notes
                
                IMPORTANT: You must respond with a valid JSON object in the following format:
                {{
                    "is_data_tab": true/false,
                    "confidence": number between 0 and 1,
                    "reason": "explanation of your decision"
                }}
                
                Do not include any other text in your response. Only return the JSON object.
                """
                
                # Get LLM response
                response = model.generate_content(prompt)
                print(f"LLM response for {sheet_name}: {response.text}")
                
                # Clean the response text
                response_text = response.text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:]  # Remove ```json
                if response_text.startswith('```'):
                    response_text = response_text[3:]  # Remove ```
                if response_text.endswith('```'):
                    response_text = response_text[:-3]  # Remove ```
                response_text = response_text.strip()
                
                # Parse the response
                result = json.loads(response_text)
                
                if result["is_data_tab"]:
                    data_tabs[sheet_name] = result["confidence"]
            
            print(f"Found data tabs: {data_tabs}")
            
            # Process only the data tabs
            cleaned_sheets = {}
            for sheet_name in data_tabs.keys():
                print(f"Processing data sheet: {sheet_name}")
                
                # Read the entire sheet
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                # Convert DataFrame to raw text for analysis
                raw_content = df.head().to_csv(index=False)
                
                # Create a prompt for the LLM to analyze the raw content
                prompt = f"""
                Analyze this raw Excel sheet content and:
                1. Identify which rows are metadata (should be removed)
                2. Identify which row contains the actual data headers
                3. Suggest proper column names if missing
                
                Sheet Name: {sheet_name}
                Content:
                {raw_content}
                
                IMPORTANT: You must respond with a valid JSON object in the following format:
                {{
                    "metadata_rows": [list of row numbers to remove (0-based)],
                    "header_row": number (row number containing headers, -1 if no headers),
                    "column_names": [list of suggested column names if headers are missing],
                    "data_start_row": number (row number where actual data begins)
                }}
                
                Do not include any other text in your response. Only return the JSON object.
                """
                
                # Get LLM response
                response = model.generate_content(prompt)
                print(f"LLM response for {sheet_name}: {response.text}")
                
                # Clean the response text
                response_text = response.text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:]  # Remove ```json
                if response_text.startswith('```'):
                    response_text = response_text[3:]  # Remove ```
                if response_text.endswith('```'):
                    response_text = response_text[:-3]  # Remove ```
                response_text = response_text.strip()
                
                # Parse the response
                result = json.loads(response_text)
                
                # Process the content based on LLM analysis
                if result["metadata_rows"]:
                    df = df.drop(result["metadata_rows"])
                
                if result["header_row"] == -1:
                    # No headers in original data, use suggested column names
                    df.columns = result["column_names"]
                else:
                    # Extract column names from the specified header row first
                    new_columns = df.iloc[result["header_row"]]

                    # Then slice from data_start_row onwards
                    df = df.iloc[result["data_start_row"]:].copy()

                    # Assign new column names
                    df.columns = new_columns

                cleaned_sheets[sheet_name] = df
            
            # Return the cleaned sheets
            if len(cleaned_sheets) == 1:
                return list(cleaned_sheets.values())[0]
            elif len(cleaned_sheets) > 1:
                return cleaned_sheets
            else:
                return None
                
        elif 'txt' in filename.lower() or 'dat' in filename.lower():
            # Try to parse as a generic text file
            # First attempt comma separator
            try:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=',')
            except:
                # Then try tab separator
                try:
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep='\t')
                except:
                    # Then try whitespace separator
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep='\s+')
            #df = clean_metadata(df)
        else:
            print(f"Unsupported file type: {filename}")
            return None
        
        print(f"Successfully read file. DataFrame shape: {df.shape}")
        return process_data(df)
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None


def process_data(df):
    """Process the uploaded data file."""
    print("Processing data...")
    print(f"Input DataFrame shape: {df.shape}")
    print(f"Input DataFrame columns: {df.columns.tolist()}")
    
    # Ensure all column names are strings
    df.columns = [str(col) for col in df.columns]
    
    # Convert column names to lowercase
    df.columns = df.columns.str.lower()
    print("Converted column names to lowercase")
    
    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    print("Handled missing values")
    
    # Convert numeric columns, excluding date/time columns
    for col in df.columns:
        # Skip if column is already numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        # Skip date/time columns
        if 'date' in col or 'time' in col:
            try:
                df[col] = pd.to_datetime(df[col])
                print(f"Converted column {col} to datetime")
            except:
                print(f"Could not convert column {col} to datetime")
        else:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"Converted column {col} to numeric")
            except:
                print(f"Could not convert column {col} to numeric")
    
    print(f"Final DataFrame shape: {df.shape}")
    print(f"Final DataFrame columns: {df.columns.tolist()}")
    return df

def clean_metadata(df):
    """
    Clean metadata from a DataFrame by:
    1. Identifying and removing metadata rows
    2. Assigning proper column names if missing
    3. Removing unnecessary metadata
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Create prompt for LLM
    print('Cleaning metadata', df.shape)
    
    # Ensure all column names are strings
    df.columns = [str(col) for col in df.columns]
    
    # Convert DataFrame to string representation, handling numeric values
    data_str = df.head(10).applymap(lambda x: str(x) if pd.notna(x) else '').to_string()
    
    prompt = f"""
    Analyze this data and:
    1. Identify if there are metadata rows that should be removed
    2. Check if proper column names are present
    3. Suggest any necessary cleaning steps
    
    Data:
    {data_str}
    
    IMPORTANT: You must respond with a valid JSON object in the following format:
    {{
        "metadata_rows": [list of row indices to remove],
        "column_names": [list of suggested column names if current ones are missing],
        "needs_cleaning": boolean,
        "reason": string
    }}

    Do not include any other text in your response. Only return the JSON object.
    """
    
    try:
        # Get LLM response
        response = model.generate_content(prompt)

        # Try to parse the response
        try:
            # Clean the response text to ensure it's valid JSON
            response_text = response.text.strip()
            if not response_text.startswith('{'):
                # Try to find the JSON object in the response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    response_text = response_text[start_idx:end_idx+1]
            
            result = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM response as JSON for {response_text}: {e}")
            # If JSON parsing fails, use rule-based result
            pass

        if result["needs_cleaning"]:
            df = df.reset_index(drop=True)

            # Remove metadata rows
            if result["metadata_rows"]:
                df = df.drop(result["metadata_rows"])
            
            # Assign new column names if needed
            if result["column_names"]:
                # Ensure all column names are strings
                new_columns = [str(col) for col in result["column_names"]]
                df.columns = new_columns
            
            # Reset index after removing rows
            df = df.reset_index(drop=True)
    
    except Exception as e:
        print(f"Error cleaning metadata: {str(e)}")
        print(f"Response text that failed to parse: {response_text}")
    
    return df 