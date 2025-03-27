import base64
import io
import pandas as pd
import numpy as np


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
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename or 'xlsx' in filename:
            print("Processing Excel file")
            df = pd.read_excel(io.BytesIO(decoded))
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
        else:
            print(f"Unsupported file type: {filename}")
            return None
        
        print(f"Successfully read file. DataFrame shape: {df.shape}")
        return process_data(df)
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None


def clean_column_name(col_name):
    """Clean and standardize column names."""
    if isinstance(col_name, str):
        # Remove special characters and replace with underscore
        import re
        col_name = re.sub(r'[^\w\s]', '_', col_name)
        # Replace multiple underscores with a single one
        col_name = re.sub(r'_+', '_', col_name)
        # Remove leading/trailing underscores
        col_name = col_name.strip('_')
        # Replace spaces with underscores
        col_name = col_name.replace(' ', '_')
        # Convert to lowercase
        col_name = col_name.lower()
        
        return col_name
    return col_name


def process_data(df):
    """Process the uploaded data file."""
    print("Processing data...")
    print(f"Input DataFrame shape: {df.shape}")
    print(f"Input DataFrame columns: {df.columns.tolist()}")
    
    # Convert column names to lowercase
    df.columns = df.columns.str.lower()
    print("Converted column names to lowercase")
    
    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    print("Handled missing values")
    
    # Convert numeric columns, excluding date/time columns
    for col in df.columns:
        if df[col].dtype == 'object':
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


def process_battery_data(df):
    """Process battery cycling data."""
    
    # Try to identify key columns
    cycle_col = next((col for col in df.columns if 'cycle' in col), None)
    voltage_col = next((col for col in df.columns if 'voltage' in col or 'v' == col), None)
    capacity_col = next((col for col in df.columns if 'capacity' in col), None)
    current_col = next((col for col in df.columns if 'current' in col or 'i' == col), None)
    
    # If key columns are missing, try to infer them from the data
    if not cycle_col and len(df.columns) >= 1:
        # Check if the first column might be cycle numbers (integers)
        if pd.api.types.is_numeric_dtype(df.iloc[:, 0]) and df.iloc[:, 0].nunique() < len(df) / 2:
            cycle_col = df.columns[0]
    
    if not capacity_col and len(df.columns) >= 2:
        # Second column might be capacity
        capacity_col = df.columns[1]
    
    if not voltage_col and len(df.columns) >= 3:
        # Third column might be voltage
        voltage_col = df.columns[2]
    
    # Calculate coulombic efficiency if charge and discharge capacity columns exist
    charge_cap_col = next((col for col in df.columns if 'charge' in col and 'capacity' in col), None)
    discharge_cap_col = next((col for col in df.columns if 'discharge' in col and 'capacity' in col), None)
    
    if charge_cap_col and discharge_cap_col:
        df['coulombic_efficiency'] = df[discharge_cap_col] / df[charge_cap_col] * 100
    
    return df


def process_xrd_data(df):
    """Process X-ray diffraction data."""
    
    # Try to identify key columns
    angle_col = next((col for col in df.columns if 'theta' in col or 'angle' in col or '2theta' in col), None)
    intensity_col = next((col for col in df.columns if 'intensity' in col or 'counts' in col), None)
    
    # If key columns are missing, assume the first column is angle (2theta) and second is intensity
    if not angle_col and len(df.columns) >= 1:
        angle_col = df.columns[0]
    
    if not intensity_col and len(df.columns) >= 2:
        intensity_col = df.columns[1]
    
    # Make sure the columns are in order (angle, intensity)
    if angle_col and intensity_col:
        # Keep only the essential columns if there are many
        if len(df.columns) > 2:
            df = df[[angle_col, intensity_col]]
        
        # Rename columns for consistency
        df = df.rename(columns={angle_col: '2theta', intensity_col: 'intensity'})
    
    return df


def process_raman_data(df):
    """Process Raman spectroscopy data."""
    
    # Try to identify key columns
    wavenumber_col = next((col for col in df.columns if 'wave' in col or 'raman' in col or 'shift' in col), None)
    intensity_col = next((col for col in df.columns if 'intensity' in col or 'counts' in col), None)
    
    # If key columns are missing, assume the first column is wavenumber and second is intensity
    if not wavenumber_col and len(df.columns) >= 1:
        wavenumber_col = df.columns[0]
    
    if not intensity_col and len(df.columns) >= 2:
        intensity_col = df.columns[1]
    
    # Make sure the columns are in order (wavenumber, intensity)
    if wavenumber_col and intensity_col:
        # Keep only the essential columns if there are many
        if len(df.columns) > 2:
            df = df[[wavenumber_col, intensity_col]]
        
        # Rename columns for consistency
        df = df.rename(columns={wavenumber_col: 'wavenumber', intensity_col: 'intensity'})
    
    return df


def process_generic_data(df):
    """Generic data processing for unidentified data types."""
    
    # Try to convert string columns to numeric if they look like numbers
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass
    
    return df 