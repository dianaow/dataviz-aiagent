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

def analyze_raw_content(content, filename, max_rows=10, last_rows=5):
    """
    Analyze raw content using LLM to identify metadata, headers, and data structure.
    
    Args:
        content (str): The raw content to analyze
        filename (str): The filename of the data file
        max_rows (int): Number of rows to show from the beginning
        last_rows (int): Number of rows to show from the end
        
    Returns:
        dict: Analysis results containing metadata rows, header row, column names, and data start row
    """
    # Split content into lines and truncate
    lines = content.split('\n')
    truncated_content = '\n'.join(lines[:max_rows] + ['...'] + lines[-last_rows:])

    # Create appropriate prompt based on content type
    prompt = f"""
    Analyze this raw data file content and:
    1. Identify which rows are metadata (should be removed). Metadata typically includes:
        - Title rows
        - Notes or comments
        - Empty rows
        - Rows with merged cells
        - Rows with only a few filled cells
        - Rows with text descriptions instead of column headers
    
    2. Identify which row contains the actual data headers. A valid header row should:
        - May not be the first row, but most likely the first non-metadata row
        - Contain unique column names all as texts
        - Have consistent formatting (no merged cells)
        - Be followed by rows with actual data

    3. Suggest proper column names if missing. The column names should:
        - Be descriptive and meaningful, reference the metadata if possible to create a meaningful column name
        - Be in lowercase
        - Use underscores instead of spaces
        - Be unique
        - Not contain special characters
        - Must match the number of columns with values in the data. If there are only two columns with values, there should be two column names.
    
    File Name: {filename}
    Content (truncated to first {max_rows} and last {last_rows} rows):
    {truncated_content}
    """
    
    # Add common JSON format requirement
    prompt += """
    IMPORTANT: You must respond with a valid JSON object in the following format:
    {
        "metadata_rows": [list of row numbers to remove (0-based)],
        "header_row": number (row number containing headers, -1 if no headers),
        "column_names": [list of suggested column names if headers are missing],
        "data_start_row": number (row number where actual data begins)
    }
    
    Do not include any other text in your response. Only return the JSON object.
    """
    
    # Get LLM response
    response = model.generate_content(prompt)

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
    return json.loads(response_text)

def parse_contents(contents, filename):
    """Parse the uploaded file contents."""
    #print(f"Parsing file: {filename}")

    if contents is None:
        #print("No contents provided")
        return None
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        if 'csv' in filename:
            #print("Processing CSV file")
            # First, get the raw content as text
            raw_content = decoded.decode('utf-8')
            
            # Analyze the content
            result = analyze_raw_content(raw_content, filename)
            #print(f"Analysis result: {result}")

            # Read CSV with appropriate parameters based on analysis
            if(result["header_row"] == 0 and not result["metadata_rows"]):
                df = pd.read_csv(io.StringIO(raw_content), header=None)

                return process_data(df)
            else:
                lines = raw_content.split('\n')
                
                # Create a set for faster metadata row lookup
                metadata_rows = set(result["metadata_rows"])
                
                # Process lines using list comprehension
                cleaned_lines = [
                    line
                    for i, line in enumerate(lines)
                    if i not in metadata_rows and (i == result["header_row"] or i >= result["data_start_row"])
                ]
                
                cleaned_content = '\n'.join(cleaned_lines)

                df = pd.read_csv(io.StringIO(cleaned_content), header=None)

                # Set column names
                if result["header_row"] != -1:
                    df.columns = df.iloc[result["header_row"]].tolist()
                    df = df.drop(result["header_row"])
                else:
                    df.columns = result["column_names"]

                return df.to_json(date_format='iso', orient='split')
        
        elif 'xls' in filename or 'xlsx' in filename:
            #print("Processing Excel file")
            # Read all sheets into a dictionary of dataframes
            excel_file = io.BytesIO(decoded)
            all_sheets = pd.read_excel(excel_file, sheet_name=None, header=None)
            
            # Process each sheet
            cleaned_sheets = {}
            
            # First, identify which tabs contain data
            data_tabs = {}
            for sheet_name, df in all_sheets.items():
                #print(f"Analyzing sheet: {sheet_name}")
                
                # Skip empty sheets
                if df.empty:
                    #print(f"Sheet {sheet_name} is empty")
                    continue
                
                # Get first 10 and last 5 rows for analysis
                sample_df = pd.concat([
                    df.head(10),
                    pd.DataFrame([['...'] * len(df.columns)], columns=df.columns),
                    df.tail(5)
                ])
                
                # Convert sample to text for analysis
                raw_content = sample_df.to_string(index=False)
                
                # Create a prompt for the LLM to analyze if this is a data tab
                prompt = f"""
                Analyze this Excel sheet and determine if it contains actual data or is just metadata.
                
                Sheet Name: {sheet_name}
                Content (first 10 rows, separator, last 5 rows):
                {raw_content}
                
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
                #print(f"LLM response for {sheet_name}: {response.text}")
                
                # Clean and parse the response
                response_text = response.text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.startswith('```'):
                    response_text = response_text[3:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                result = json.loads(response_text)
                
                if result["is_data_tab"]:
                    data_tabs[sheet_name] = result["confidence"]
            
            #print(f"Found data tabs: {data_tabs}")
            
            # Process only the data tabs
            for sheet_name in data_tabs.keys():
                #print(f"Processing data sheet: {sheet_name}")
                df = all_sheets[sheet_name]
                
                # Get sample for analysis
                sample_df = pd.concat([
                    df.head(10),
                    pd.DataFrame([['...'] * len(df.columns)], columns=df.columns),
                    df.tail(5)
                ])
                raw_content = sample_df.to_string(index=False)
                
                # Analyze the content
                result = analyze_raw_content(raw_content, filename)
                #print(f"Analysis result: {result}")
                try:
                    # Set column names
                    if result["header_row"] != -1:
                        df.columns = df.iloc[result["header_row"]].tolist()
                    else:
                        df.columns = result["column_names"]
                    
                    # Filter out metadata rows
                    if result["metadata_rows"]:
                        df = df.drop(result["metadata_rows"])

                    if result["header_row"] != -1:
                        df = df.drop(result["header_row"])
                        
                    # Reset index after all row removals
                    df = df.reset_index(drop=True)

                except Exception as e:
                    #print(f"Error processing Excel: {str(e)}")
                    return None
                
                cleaned_sheets[sheet_name] = df
                #print(f"Completed processing sheet: {sheet_name}")
            
            # Return the cleaned sheets
            if len(cleaned_sheets) == 1:
                # Convert single DataFrame to JSON
                df = list(cleaned_sheets.values())[0]
                return df.to_json(date_format='iso', orient='split')
            elif len(cleaned_sheets) > 1:
                # Convert dictionary of DataFrames to dictionary of JSON strings
                return {sheet_name: df.to_json(date_format='iso', orient='split') 
                       for sheet_name, df in cleaned_sheets.items()}
            else:
                return None
                
        elif 'txt' in filename.lower() or 'dat' in filename.lower():
            #print("Processing text file")
            # First, get the raw content as text
            raw_content = decoded.decode('utf-8')
            
            # Analyze the content
            result = analyze_raw_content(raw_content, filename)
            
            # Process the content based on LLM analysis
            lines = raw_content.split('\n')
            cleaned_lines = []
            for i, line in enumerate(lines):
                if i not in result["metadata_rows"]:
                    if i == result["header_row"] or (result["header_row"] == -1 and i == result["data_start_row"]):
                        # This is either the header line or the first data line
                        if result["header_row"] == -1:
                            # No headers, use suggested column names
                            cleaned_lines.append(','.join(result["column_names"]))
                        cleaned_lines.append(line)
                    elif i >= result["data_start_row"]:
                        # This is a data line
                        cleaned_lines.append(line)
            
            # Join the cleaned lines back together
            cleaned_content = '\n'.join(cleaned_lines)
            
            # Try different separators
            try:
                df = pd.read_csv(io.StringIO(cleaned_content), sep=',')
            except:
                try:
                    df = pd.read_csv(io.StringIO(cleaned_content), sep='\t')
                except:
                    df = pd.read_csv(io.StringIO(cleaned_content), sep='\s+')
            
            # Process and convert to JSON
            #df = process_data(df)
            return df.to_json(date_format='iso', orient='split')
        
        else:
            #print(f"Unsupported file type: {filename}")
            return None
        
    except Exception as e:
        #print(f"Error processing file: {str(e)}")
        return None


def process_data(df):
    """Process the uploaded data file."""
    # Convert column names to lowercase
    df.columns = df.columns.str.lower()

    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')

    # Convert numeric columns, excluding date/time columns
    for col in df.columns:
        if df[col].dtype == 'object':
            if 'date' in col or 'time' in col:
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
            else:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass

    return df