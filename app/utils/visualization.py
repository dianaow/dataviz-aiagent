import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import find_peaks
from scipy import optimize


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