import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import find_peaks
from scipy import optimize
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv
import dash.html as html

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Gemini model
model = genai.GenerativeModel('gemini-2.0-flash')

def generate_visualizations(df, data_type, plot_type=None):
    """
    Generate visualizations based on the data type and selected plot type.
    
    Args:
        df (pd.DataFrame): The data to visualize
        data_type (str): The type of data (e.g., "Battery Cycling Data")
        plot_type (str, optional): Specific plot type to generate
        
    Returns:
        tuple: (primary_fig, secondary_fig)
    """
    if data_type == "Battery Cycling Data":
        return generate_battery_visualizations(df, plot_type)
    elif data_type == "X-ray Diffraction":
        return generate_xrd_visualizations(df, plot_type)
    elif data_type == "Raman Spectroscopy":
        return generate_raman_visualizations(df, plot_type)
    else:
        return generate_generic_visualizations(df, plot_type)


def generate_battery_visualizations(df, plot_type=None):
    """Generate visualizations for battery cycling data."""
    
    # Try to identify key columns
    cycle_col = next((col for col in df.columns if 'cycle' in col), None)
    voltage_col = next((col for col in df.columns if 'voltage' in col or 'v' == col), None)
    capacity_col = next((col for col in df.columns if 'capacity' in col), None)
    current_col = next((col for col in df.columns if 'current' in col or 'i' == col), None)
    time_col = next((col for col in df.columns if 'time' in col or 'data' == col), None)

    # Default to first numeric column if we can't find specific columns
    if not cycle_col:
        cycle_col = next((col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])), df.columns[0])
    
    if not capacity_col:
        for col in df.columns:
            if col != cycle_col and pd.api.types.is_numeric_dtype(df[col]):
                capacity_col = col
                break
    
    # Create primary figure based on plot type
    if not plot_type or plot_type == "capacity_cycle":
      # Capacity vs Cycle plot
      fig1 = go.Figure()
      
      df["Cycle_Capacity(Ah)"] = df[current_col] * (df[time_col] / 3600)  # Convert seconds to hours
      cycle_capacity_calc = df.groupby(cycle_col, as_index=False)["Cycle_Capacity(Ah)"].sum()

      if cycle_col:
          fig1.add_trace(go.Scatter(
              x=cycle_capacity_calc[cycle_col],
              y=cycle_capacity_calc["Cycle_Capacity(Ah)"],
              mode='lines+markers',
              name='Cycle Capacity',
              marker=dict(size=8)
          ))
          
          fig1.update_layout(
              title="Capacity vs. Cycle Number",
              xaxis_title="Cycle Number",
              yaxis_title="Capacity (Ah)",
              template="plotly_white"
          )
      else:
          fig1 = default_figure("Missing required columns for Capacity vs. Cycle plot")
  
    elif plot_type == "voltage_time":
      # Voltage vs Time plot
      fig1 = go.Figure()
      
      if voltage_col and time_col:
          fig1.add_trace(go.Scatter(
              x=df[time_col],
              y=df[voltage_col],
              mode='lines',
              name='Voltage'
          ))
          
          fig1.update_layout(
              title="Battery Voltage vs. Time",
              xaxis_title="Time (s)",
              yaxis_title="Voltage (V)",
              template="plotly_white"
          )
      else:
          fig1 = default_figure("Missing required columns for Battery Voltage vs. Time plot")
    
    elif plot_type == "current_time":
      # Currentvs Time plot
      fig1 = go.Figure()
      
      if current_col and capacity_col:
          fig1.add_trace(go.Scatter(
              x=df[time_col],
              y=df[current_col],
              mode='lines',
              name='Current'
          ))
          
          fig1.update_layout(
              title="Battery Current vs. Time",
              xaxis_title="Time (s)",
              yaxis_title="Current (A)",
              template="plotly_white"
          )
      else:
          fig1 = default_figure("Missing required columns for Battery Current vs. Time plot")
    
    elif plot_type == "retention_cycle":
      # Capacity Retention vs Cycle Number
      fig1 = go.Figure()
      
      df["Cycle_Capacity(Ah)"] = df[current_col] * (df[time_col] / 3600)  # Convert seconds to hours
      cycle_capacity_calc = df.groupby(cycle_col, as_index=False)["Cycle_Capacity(Ah)"].sum()
      reference_cycle = 2
      reference_capacity = cycle_capacity_calc.loc[
          cycle_capacity_calc[cycle_col] == reference_cycle, "Cycle_Capacity(Ah)"
      ].values[0]
      cycle_capacity_calc["Capacity_Retention(%)"] = (
          cycle_capacity_calc["Cycle_Capacity(Ah)"] / reference_capacity
      ) * 100

      if current_col and capacity_col:
          fig1.add_trace(go.Scatter(
              x=cycle_capacity_calc[cycle_col],
              y=cycle_capacity_calc["Cycle_Capacity(Ah)"],
              mode='lines',
              name='Capacity Retention'
          ))
          
          fig1.update_layout(
              title="Capacity Retention vs. Cycle Number (Relative to 2nd Cycle)",
              xaxis_title="Cycle Number",
              yaxis_title="Capacity Retention (%)",
              template="plotly_white"
          )
      else:
          fig1 = default_figure("Missing required columns for Capacity Retention vs. Cycle Number plot")
    
    else:
        # Default to capacity vs cycle
        fig1 = go.Figure()
        
        if cycle_col and capacity_col:
            fig1.add_trace(go.Scatter(
                x=df[cycle_col],
                y=df[capacity_col],
                mode='lines+markers',
                name='Capacity',
                marker=dict(size=8)
            ))
            
            fig1.update_layout(
                title="Capacity vs. Cycle Number",
                xaxis_title="Cycle Number",
                yaxis_title="Capacity (mAh/g)",
                template="plotly_white"
            )
        else:
            fig1 = default_figure("Missing required columns for default plot")
    
    # Create secondary figure - differential capacity analysis
    fig2 = None
    
    if voltage_col and capacity_col:
        # Sort by voltage to ensure proper differentiation
        sorted_df = df.sort_values(by=voltage_col)
        
        # Calculate differential capacity (dQ/dV)
        # Use numpy's gradient function for better numerical differentiation
        sorted_df = sorted_df.copy()
        sorted_df['dQ_dV'] = np.gradient(sorted_df[capacity_col], sorted_df[voltage_col])
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=sorted_df[voltage_col],
            y=sorted_df['dQ_dV'],
            mode='lines',
            name='dQ/dV'
        ))
        
        fig2.update_layout(
            title="Differential Capacity Analysis (dQ/dV)",
            xaxis_title="Voltage (V)",
            yaxis_title="dQ/dV",
            template="plotly_white"
        )
    
    return fig1, fig2


def generate_xrd_visualizations(df, plot_type=None):
    """Generate visualizations for X-ray diffraction data."""
    
    # Try to identify key columns
    angle_col = next((col for col in df.columns if 'theta' in col or 'angle' in col or '2theta' in col), None)
    intensity_col = next((col for col in df.columns if 'intensity' in col or 'counts' in col), None)
    
    # If key columns are missing, assume the first two columns
    if not angle_col and len(df.columns) >= 1:
        angle_col = df.columns[0]
    
    if not intensity_col and len(df.columns) >= 2:
        intensity_col = df.columns[1]
    
    # Make sure we're using consistent column names
    if angle_col != '2theta' and intensity_col != 'intensity':
        df = df.rename(columns={angle_col: '2theta', intensity_col: 'intensity'})
        angle_col = '2theta'
        intensity_col = 'intensity'
    
    # Create primary figure based on plot type
    if not plot_type or plot_type == "diffraction_pattern":
        # Basic diffraction pattern
        fig1 = go.Figure()
        
        fig1.add_trace(go.Scatter(
            x=df[angle_col],
            y=df[intensity_col],
            mode='lines',
            name='XRD Pattern',
            line=dict(color='blue')
        ))
        
        fig1.update_layout(
            title="X-ray Diffraction Pattern",
            xaxis_title="2θ (degrees)",
            yaxis_title="Intensity (counts)",
            template="plotly_white"
        )
    
    else:
        # Default to basic diffraction pattern
        fig1 = go.Figure()
        
        fig1.add_trace(go.Scatter(
            x=df[angle_col],
            y=df[intensity_col],
            mode='lines',
            name='XRD Pattern',
            line=dict(color='blue')
        ))
        
        fig1.update_layout(
            title="X-ray Diffraction Pattern",
            xaxis_title="2θ (degrees)",
            yaxis_title="Intensity (counts)",
            template="plotly_white"
        )
    
    # Create secondary figure - peak fitting
    fig2 = go.Figure()
    
    try:
        # Find peaks
        peak_indices, _ = find_peaks(df[intensity_col], height=0.1*df[intensity_col].max(), 
                                    distance=10, prominence=0.2*df[intensity_col].max())
        
        # Plot original data
        fig2.add_trace(go.Scatter(
            x=df[angle_col],
            y=df[intensity_col],
            mode='lines',
            name='XRD Pattern',
            line=dict(color='lightblue')
        ))
        
        # Gaussian function for fitting
        def gaussian(x, amp, cen, wid):
            return amp * np.exp(-(x - cen)**2 / (2 * wid**2))
        
        # Only fit a few peaks for simplicity
        if len(peak_indices) > 0:
            selected_peaks = peak_indices[:min(3, len(peak_indices))]
            
            # Fit a Gaussian to each peak and plot
            for i, peak_idx in enumerate(selected_peaks):
                peak_pos = df[angle_col].iloc[peak_idx]
                peak_height = df[intensity_col].iloc[peak_idx]
                
                # Select data around peak for fitting
                window = int(len(df) / 20)  # Adjust window size based on data
                start_idx = max(0, peak_idx - window)
                end_idx = min(len(df), peak_idx + window)
                
                x_data = df[angle_col].iloc[start_idx:end_idx]
                y_data = df[intensity_col].iloc[start_idx:end_idx]
                
                # Initial guess for Gaussian parameters
                initial_guess = [peak_height, peak_pos, 0.2]
                
                try:
                    # Fit the Gaussian
                    popt, _ = optimize.curve_fit(gaussian, x_data, y_data, p0=initial_guess)
                    
                    # Plot the fitted Gaussian
                    x_fit = np.linspace(min(x_data), max(x_data), 100)
                    y_fit = gaussian(x_fit, *popt)
                    
                    fig2.add_trace(go.Scatter(
                        x=x_fit,
                        y=y_fit,
                        mode='lines',
                        name=f'Peak {i+1} (FWHM: {2.355*popt[2]:.2f}°)',
                        line=dict(color=['red', 'green', 'purple'][i % 3])
                    ))
                    
                    # Add annotation with peak parameters
                    fig2.add_annotation(
                        x=popt[1],
                        y=popt[0],
                        text=f"Peak: {popt[1]:.2f}°<br>FWHM: {2.355*popt[2]:.2f}°",
                        showarrow=True,
                        arrowhead=1,
                        ax=0,
                        ay=-60
                    )
                except:
                    continue
        
        fig2.update_layout(
            title="XRD Peak Fitting",
            xaxis_title="2θ (degrees)",
            yaxis_title="Intensity (counts)",
            template="plotly_white"
        )
    except Exception as e:
        print(f"Error in peak fitting: {e}")
        fig2 = default_figure("Error performing peak fitting")
    
    return fig1, fig2


def generate_raman_visualizations(df, plot_type=None):
    """Generate visualizations for Raman spectroscopy data."""
    
    # Try to identify key columns
    wavenumber_col = next((col for col in df.columns if 'wave' in col or 'raman' in col or 'shift' in col), None)
    intensity_col = next((col for col in df.columns if 'intensity' in col or 'counts' in col), None)
    
    # If key columns are missing, assume the first two columns
    if not wavenumber_col and len(df.columns) >= 1:
        wavenumber_col = df.columns[0]
    
    if not intensity_col and len(df.columns) >= 2:
        intensity_col = df.columns[1]
    
    # Make sure we're using consistent column names
    if wavenumber_col != 'wavenumber' and intensity_col != 'intensity':
        df = df.rename(columns={wavenumber_col: 'wavenumber', intensity_col: 'intensity'})
        wavenumber_col = 'wavenumber'
        intensity_col = 'intensity'
    
    # Create primary figure based on plot type
    if not plot_type or plot_type == "raman_spectrum":
        # Basic Raman spectrum
        fig1 = go.Figure()
        
        fig1.add_trace(go.Scatter(
            x=df[wavenumber_col],
            y=df[intensity_col],
            mode='lines',
            name='Raman Spectrum',
            line=dict(color='blue')
        ))
        
        fig1.update_layout(
            title="Raman Spectroscopy",
            xaxis_title="Wavenumber (cm⁻¹)",
            yaxis_title="Intensity (a.u.)",
            template="plotly_white"
        )
    
    else:
        # Default to basic Raman spectrum
        fig1 = go.Figure()
        
        fig1.add_trace(go.Scatter(
            x=df[wavenumber_col],
            y=df[intensity_col],
            mode='lines',
            name='Raman Spectrum',
            line=dict(color='blue')
        ))
        
        fig1.update_layout(
            title="Raman Spectroscopy",
            xaxis_title="Wavenumber (cm⁻¹)",
            yaxis_title="Intensity (a.u.)",
            template="plotly_white"
        )
    
    # Create secondary figure - peak deconvolution
    fig2 = go.Figure()
    
    try:
        # Find peaks
        peak_indices, _ = find_peaks(df[intensity_col], height=0.2*df[intensity_col].max(), 
                                    distance=20, prominence=0.1*df[intensity_col].max())
        
        # Plot original data
        fig2.add_trace(go.Scatter(
            x=df[wavenumber_col],
            y=df[intensity_col],
            mode='lines',
            name='Raman Spectrum',
            line=dict(color='lightblue')
        ))
        
        # Lorentzian function for fitting Raman peaks
        def lorentzian(x, amp, cen, wid):
            return amp * wid**2 / ((x - cen)**2 + wid**2)
        
        # Only fit a few peaks for simplicity
        if len(peak_indices) > 0:
            selected_peaks = peak_indices[:min(3, len(peak_indices))]
            sum_fitted = np.zeros(len(df))
            
            # Fit a Lorentzian to each peak and plot
            for i, peak_idx in enumerate(selected_peaks):
                peak_pos = df[wavenumber_col].iloc[peak_idx]
                peak_height = df[intensity_col].iloc[peak_idx]
                
                # Select data around peak for fitting
                window = int(len(df) / 15)  # Adjust window size based on data
                start_idx = max(0, peak_idx - window)
                end_idx = min(len(df), peak_idx + window)
                
                x_data = df[wavenumber_col].iloc[start_idx:end_idx]
                y_data = df[intensity_col].iloc[start_idx:end_idx]
                
                # Initial guess for Lorentzian parameters
                initial_guess = [peak_height, peak_pos, 10]
                
                try:
                    # Fit the Lorentzian
                    popt, _ = optimize.curve_fit(lorentzian, x_data, y_data, p0=initial_guess)
                    
                    # Plot the fitted Lorentzian
                    y_fit = lorentzian(df[wavenumber_col], *popt)
                    sum_fitted += y_fit
                    
                    fig2.add_trace(go.Scatter(
                        x=df[wavenumber_col],
                        y=y_fit,
                        mode='lines',
                        name=f'Peak {i+1} ({popt[1]:.1f} cm⁻¹)',
                        line=dict(color=['red', 'green', 'purple'][i % 3])
                    ))
                except:
                    continue
            
            # Plot sum of all fitted peaks
            if np.any(sum_fitted > 0):
                fig2.add_trace(go.Scatter(
                    x=df[wavenumber_col],
                    y=sum_fitted,
                    mode='lines',
                    name='Sum of Fitted Peaks',
                    line=dict(color='black', dash='dash')
                ))
        
        fig2.update_layout(
            title="Raman Peak Deconvolution",
            xaxis_title="Wavenumber (cm⁻¹)",
            yaxis_title="Intensity (a.u.)",
            template="plotly_white"
        )
    except Exception as e:
        print(f"Error in Raman peak fitting: {e}")
        fig2 = default_figure("Error performing Raman peak deconvolution")
    
    return fig1, fig2


def generate_generic_visualizations(df, plot_type=None):
    """Generate visualizations for generic data types."""
    
    # Try to get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        # Not enough numeric columns for plotting
        fig1 = default_figure("Not enough numeric columns for plotting")
        return fig1, None
    
    # Create primary figure based on plot type
    if not plot_type or plot_type == "standard":
        # Create a line plot of all numeric columns
        fig1 = go.Figure()
        
        for col in numeric_cols[:5]:  # Limit to first 5 columns to avoid overcrowding
            fig1.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                mode='lines',
                name=col
            ))
        
        fig1.update_layout(
            title="Standard Line Plot",
            xaxis_title="Index",
            yaxis_title="Value",
            template="plotly_white"
        )
    
    elif plot_type == "scatter":
        # Create a scatter plot of first two numeric columns
        fig1 = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                        title=f"Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}")
        
        fig1.update_layout(
            template="plotly_white"
        )
    
    elif plot_type == "line":
        # Create a line plot of first numeric column
        fig1 = px.line(df, y=numeric_cols[0], title=f"Line Plot: {numeric_cols[0]}")
        
        fig1.update_layout(
            template="plotly_white"
        )
    
    else:
        # Default to line plot
        fig1 = px.line(df, y=numeric_cols[0], title=f"Line Plot: {numeric_cols[0]}")
        
        fig1.update_layout(
            template="plotly_white"
        )
    
    # Create secondary figure - correlation heatmap
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        
        fig2 = px.imshow(corr_matrix, 
                      color_continuous_scale='RdBu_r',
                      zmin=-1, zmax=1, 
                      title="Correlation Matrix")
        
        fig2.update_layout(
            template="plotly_white"
        )
    else:
        fig2 = None
    
    return fig1, fig2


def default_figure(message):
    """Create a default figure with a message."""
    fig = go.Figure()
    
    fig.add_annotation(
        x=0.5, y=0.5,
        text=message,
        font=dict(size=14),
        showarrow=False,
        xref="paper", yref="paper"
    )
    
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig


def generate_llm_visualizations(df, data_type, error_message=None):
    """
    Generate multiple visualization options using LLM with fallback to rule-based visualization.
    
    Args:
        df (pd.DataFrame): The data to visualize
        data_type (str): The type of data (e.g., "Battery Cycling Data")
        error_message (str, optional): Error message from previous attempt
        
    Returns:
        list: List of Plotly figures
    """
    try:
        # Create a sample of the data
        data_sample = df.copy()
        for col in data_sample.columns:
            if pd.api.types.is_datetime64_any_dtype(data_sample[col]):
                data_sample[col] = data_sample[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        data_sample = data_sample.head(5).to_dict(orient='records')
        
        # Prepare the prompt for Gemini
        prompt = f"""You are an expert in data visualization and materials science.
        Your task is to generate multiple Plotly visualization options for the provided data.
        
        Data type: {data_type}
        Available columns: {df.columns.tolist()}
        Data sample: {json.dumps(data_sample, indent=2)}
        
        {f'Previous error: {error_message}' if error_message else ''}
        
        Generate multiple visualization options that would be useful for this data type.

        IMPORTANT: Respond with ONLY a JSON object containing the visualization options.
        Do not include any other text or explanations.

        For the x and y attributes, specify the column names from the available columns list. 
        Perform any necessary calculations and create new columns if needed to plot the data.
        Do not include the actual data arrays.

        The response should be in this exact format:
        {{
            "visualization_options": [
                {{
                    "name": "Descriptive Name 1",
                    "description": "Brief description of this visualization",
                    "figure_spec": {{
                        "data": [
                            {{
                                "type": "scatter",
                                "x": "column_name_for_x",
                                "y": "column_name_for_y",
                                "mode": "lines",
                                "name": "Data"
                            }}
                        ],
                        "layout": {{
                            "title": "Plot Title",
                            "xaxis": {{"title": "X Axis"}},
                            "yaxis": {{"title": "Y Axis"}},
                            "template": "plotly_white"
                        }}
                    }}
                }},
                {{
                    "name": "Descriptive Name 2",
                    "description": "Brief description of this visualization",
                    "figure_spec": {{...}}
                }}
            ]
        }}
        
        Use the following Plotly components:
        - go.Figure() for creating figures
        - go.Scatter() for line/scatter plots
        - go.Bar() for bar plots
        - go.Histogram() for histograms
        - go.Box() for box plots
        - go.Heatmap() for correlation matrices
        
        Make sure to:
        1. Generate 3-5 different visualization options
        2. Use appropriate axis labels and titles
        3. Include legends
        4. Use the 'plotly_white' template
        5. Handle missing data appropriately
        6. Use appropriate colors and markers
        """
        
        # Call Gemini API
        response = model.generate_content(prompt)
        
        # Print the raw response for debugging
        print("Raw LLM response:", response.text)
        
        try:
            # Clean the response text to ensure it's valid JSON
            response_text = response.text.strip()
            # Remove any markdown code block markers if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse the response
            result = json.loads(response_text)
            
            # Validate the result structure
            if not isinstance(result, dict) or "visualization_options" not in result:
                raise ValueError("Response missing required 'visualization_options' key")
            
            # Create figures for each visualization option
            figures = []
            for option in result["visualization_options"]:
                # Create the figure with actual data from DataFrame
                fig = go.Figure()
                
                # Add traces using column names
                for trace in option["figure_spec"]["data"]:
                    # Create a copy of the trace specification
                    trace_spec = trace.copy()
                    
                    # Replace column names with actual data
                    if "x" in trace_spec and isinstance(trace_spec["x"], str):
                        trace_spec["x"] = df[trace_spec["x"]]
                    if "y" in trace_spec and isinstance(trace_spec["y"], str):
                        trace_spec["y"] = df[trace_spec["y"]]
                    
                    # Add the trace to the figure
                    fig.add_trace(go.Scatter(**trace_spec))
                
                # Set the layout
                fig.update_layout(**option["figure_spec"]["layout"])
                
                # Add the figure to the list
                figures.append(fig)
            
            return figures
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing LLM response: {e}")
            print("Cleaned response text:", response_text)
            if not error_message:  # Only retry once
                return generate_llm_visualizations(df, data_type, str(e))
            else:
                # Fallback to rule-based visualization
                primary_fig, secondary_fig = generate_visualizations(df, data_type)
                figures = [primary_fig]
                if secondary_fig:
                    figures.append(secondary_fig)
                return figures
                
    except Exception as e:
        print(f"Error in LLM visualization generation: {e}")
        if not error_message:  # Only retry once
            return generate_llm_visualizations(df, data_type, str(e))
        else:
            # Fallback to rule-based visualization
            primary_fig, secondary_fig = generate_visualizations(df, data_type)
            figures = [primary_fig]
            if secondary_fig:
                figures.append(secondary_fig)
            return figures 