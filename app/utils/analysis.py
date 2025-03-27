import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy import optimize
from dash import html


def analyze_data(df, data_type):
    """
    Analyze data based on its type and generate insights.
    
    Args:
        df (pd.DataFrame): The data to analyze
        data_type (str): The type of data (e.g., "Battery Cycling Data")
        
    Returns:
        tuple: (analysis_results_component, analysis_data_dict)
    """
    if data_type == "Battery Cycling Data":
        return analyze_battery_data(df)
    elif data_type == "X-ray Diffraction":
        return analyze_xrd_data(df)
    elif data_type == "Raman Spectroscopy":
        return analyze_raman_data(df)
    else:
        return analyze_generic_data(df)


def analyze_battery_data(df):
    """Analyze battery cycling data."""
    insights = []
    analysis_data = {}
    
    # Try to identify key columns
    cycle_col = next((col for col in df.columns if 'cycle' in col), None)
    voltage_col = next((col for col in df.columns if 'voltage' in col or 'v' == col), None)
    capacity_col = next((col for col in df.columns if 'capacity' in col), None)
    
    if not cycle_col or not capacity_col:
        insights.append(html.P("Unable to perform detailed analysis: missing cycle or capacity data.", className="text-danger"))
        return html.Div(insights), analysis_data
    
    # Calculate capacity retention
    try:
        initial_capacity = df[df[cycle_col] == df[cycle_col].min()][capacity_col].max()
        final_capacity = df[df[cycle_col] == df[cycle_col].max()][capacity_col].max()
        
        if initial_capacity > 0:
            retention = (final_capacity / initial_capacity) * 100
            
            insights.append(html.H5("Capacity Retention Analysis"))
            insights.append(html.P(f"Initial Capacity: {initial_capacity:.2f} mAh/g"))
            insights.append(html.P(f"Final Capacity: {final_capacity:.2f} mAh/g"))
            insights.append(html.P(f"Capacity Retention: {retention:.1f}%"))
            
            # Add analysis data
            analysis_data["capacity_retention"] = float(retention)
            analysis_data["initial_capacity"] = float(initial_capacity)
            analysis_data["final_capacity"] = float(final_capacity)
            
            # Provide context about the retention
            if retention > 80:
                insights.append(html.P("The battery shows excellent capacity retention (>80%).", 
                                       className="text-success font-weight-bold"))
            elif retention > 60:
                insights.append(html.P("The battery shows good capacity retention (60-80%).", 
                                       className="text-primary"))
            else:
                insights.append(html.P("The battery shows significant capacity fade (<60%).", 
                                       className="text-warning"))
    except Exception as e:
        print(f"Error calculating capacity retention: {e}")
    
    # Calculate capacity fade rate
    try:
        cycles = df[cycle_col].nunique()
        if cycles > 1:
            fade_rate = (initial_capacity - final_capacity) / (cycles - 1) / initial_capacity * 100
            
            insights.append(html.H5("Capacity Fade Analysis"))
            insights.append(html.P(f"Capacity Fade Rate: {fade_rate:.4f}% per cycle"))
            
            # Add analysis data
            analysis_data["fade_rate"] = float(fade_rate)
            analysis_data["cycles"] = int(cycles)
            
            # Provide context about the fade rate
            if fade_rate < 0.05:
                insights.append(html.P("The battery shows very low capacity fade rate (<0.05% per cycle).", 
                                       className="text-success font-weight-bold"))
            elif fade_rate < 0.1:
                insights.append(html.P("The battery shows good stability (fade rate <0.1% per cycle).", 
                                       className="text-primary"))
            else:
                insights.append(html.P("The battery shows high capacity fade rate (>0.1% per cycle).", 
                                       className="text-warning"))
    except Exception as e:
        print(f"Error calculating fade rate: {e}")
    
    # Calculate Coulombic efficiency statistics
    if "coulombic_efficiency" in df.columns:
        try:
            avg_ce = df["coulombic_efficiency"].mean()
            min_ce = df["coulombic_efficiency"].min()
            
            insights.append(html.H5("Coulombic Efficiency Analysis"))
            insights.append(html.P(f"Average Coulombic Efficiency: {avg_ce:.2f}%"))
            insights.append(html.P(f"Minimum Coulombic Efficiency: {min_ce:.2f}%"))
            
            # Add analysis data
            analysis_data["avg_ce"] = float(avg_ce)
            analysis_data["min_ce"] = float(min_ce)
            
            # Provide context about the Coulombic efficiency
            if avg_ce > 99.5:
                insights.append(html.P("The battery shows excellent Coulombic efficiency (>99.5%).", 
                                       className="text-success font-weight-bold"))
            elif avg_ce > 99:
                insights.append(html.P("The battery shows good Coulombic efficiency (>99%).", 
                                       className="text-primary"))
            else:
                insights.append(html.P("The battery shows sub-optimal Coulombic efficiency (<99%).", 
                                       className="text-warning"))
        except Exception as e:
            print(f"Error calculating Coulombic efficiency statistics: {e}")
    
    # Voltage analysis (if voltage data available)
    if voltage_col:
        try:
            max_voltage = df[voltage_col].max()
            min_voltage = df[voltage_col].min()
            voltage_window = max_voltage - min_voltage
            
            insights.append(html.H5("Voltage Analysis"))
            insights.append(html.P(f"Voltage Window: {voltage_window:.2f}V ({min_voltage:.2f}V - {max_voltage:.2f}V)"))
            
            # Add analysis data
            analysis_data["voltage_window"] = float(voltage_window)
            analysis_data["min_voltage"] = float(min_voltage)
            analysis_data["max_voltage"] = float(max_voltage)
        except Exception as e:
            print(f"Error calculating voltage statistics: {e}")
    
    # If no insights were generated
    if not insights:
        insights.append(html.P("No detailed analysis could be performed with the available data."))
    
    return html.Div(insights), analysis_data


def analyze_xrd_data(df):
    """Analyze X-ray diffraction data."""
    insights = []
    analysis_data = {}
    
    # Try to identify key columns
    angle_col = next((col for col in df.columns if 'theta' in col or 'angle' in col or '2theta' in col), None)
    intensity_col = next((col for col in df.columns if 'intensity' in col or 'counts' in col), None)
    
    # If key columns are missing, assume the first two columns
    if not angle_col and len(df.columns) >= 1:
        angle_col = df.columns[0]
    
    if not intensity_col and len(df.columns) >= 2:
        intensity_col = df.columns[1]
    
    # Ensure we're using consistent names
    df = df.rename(columns={angle_col: '2theta', intensity_col: 'intensity'})
    angle_col = '2theta'
    intensity_col = 'intensity'
    
    # Perform peak finding and analysis
    try:
        # Find peaks
        peak_indices, peak_properties = find_peaks(df[intensity_col], height=0.1*df[intensity_col].max(),
                                                 distance=10, prominence=0.2*df[intensity_col].max())
        
        peak_positions = df[angle_col].iloc[peak_indices]
        peak_heights = df[intensity_col].iloc[peak_indices]
        
        # Analyze peak properties
        insights.append(html.H5("XRD Peak Analysis"))
        insights.append(html.P(f"Number of significant peaks detected: {len(peak_positions)}"))
        
        # Add analysis data
        analysis_data["num_peaks"] = len(peak_positions)
        analysis_data["peak_positions"] = peak_positions.tolist()
        analysis_data["peak_intensities"] = peak_heights.tolist()
        
        # List the most intense peaks
        top_peaks = sorted(zip(peak_positions, peak_heights), key=lambda x: x[1], reverse=True)[:5]
        
        insights.append(html.P("Top peaks detected:"))
        peaks_list = []
        for i, (pos, height) in enumerate(top_peaks):
            peaks_list.append(html.Li(f"Peak at {pos:.2f}° (Intensity: {height:.0f})"))
        
        insights.append(html.Ul(peaks_list))
        
        # Calculate crystallite size using Scherrer equation (estimate)
        # D = K*λ/(β*cosθ)
        # Where K is the Scherrer constant (~0.9), λ is the X-ray wavelength (assume Cu Kα = 1.5406 Å),
        # β is the peak width (FWHM) in radians, and θ is the Bragg angle
        
        # Only use the most intense peak for this estimate
        if len(top_peaks) > 0:
            main_peak_pos = top_peaks[0][0]
            main_peak_idx = peak_indices[list(peak_positions).index(main_peak_pos)]
            
            # Get data around the peak for fitting
            window = int(len(df) / 20)
            start_idx = max(0, main_peak_idx - window)
            end_idx = min(len(df), main_peak_idx + window)
            
            x_data = df[angle_col].iloc[start_idx:end_idx]
            y_data = df[intensity_col].iloc[start_idx:end_idx]
            
            # Fit a Gaussian to estimate FWHM
            def gaussian(x, amp, cen, wid):
                return amp * np.exp(-(x - cen)**2 / (2 * wid**2))
            
            # Initial guess for Gaussian parameters
            initial_guess = [top_peaks[0][1], main_peak_pos, 0.2]
            
            try:
                # Fit the Gaussian
                popt, _ = optimize.curve_fit(gaussian, x_data, y_data, p0=initial_guess)
                
                # Calculate FWHM (2.355 * sigma for a Gaussian)
                fwhm = 2.355 * popt[2]
                
                # Convert to radians
                fwhm_rad = np.radians(fwhm)
                theta_rad = np.radians(main_peak_pos) / 2  # Convert 2θ to θ
                
                # Scherrer equation
                k = 0.9  # Scherrer constant
                wavelength = 1.5406  # Cu Kα in Å
                
                # Calculate crystallite size in Å (avoid division by zero)
                if fwhm_rad > 0 and np.cos(theta_rad) > 0:
                    crystallite_size = k * wavelength / (fwhm_rad * np.cos(theta_rad))
                    
                    # Convert to nm
                    crystallite_size_nm = crystallite_size / 10
                    
                    insights.append(html.H5("Crystallite Size Estimation"))
                    insights.append(html.P(f"Estimated crystallite size: {crystallite_size_nm:.1f} nm"))
                    insights.append(html.P("(Based on Scherrer equation using the most intense peak)"))
                    
                    # Add analysis data
                    analysis_data["crystallite_size_nm"] = float(crystallite_size_nm)
                    analysis_data["main_peak_pos"] = float(main_peak_pos)
                    analysis_data["main_peak_fwhm"] = float(fwhm)
                    
                    # Provide context about the crystallite size
                    if crystallite_size_nm < 10:
                        insights.append(html.P("The material has nano-sized crystallites (<10 nm).", 
                                             className="text-info"))
                    elif crystallite_size_nm < 100:
                        insights.append(html.P("The material has typical crystallite size for many nanomaterials.", 
                                             className="text-info"))
                    else:
                        insights.append(html.P("The material has relatively large crystallites (>100 nm).", 
                                             className="text-info"))
            except Exception as e:
                print(f"Error calculating crystallite size: {e}")
    
    except Exception as e:
        print(f"Error in XRD peak analysis: {e}")
        insights.append(html.P("Unable to perform detailed peak analysis on the provided data.", 
                             className="text-danger"))
    
    # If no insights were generated
    if not insights:
        insights.append(html.P("No detailed analysis could be performed with the available data."))
    
    return html.Div(insights), analysis_data


def analyze_raman_data(df):
    """Analyze Raman spectroscopy data."""
    insights = []
    analysis_data = {}
    
    # Try to identify key columns
    wavenumber_col = next((col for col in df.columns if 'wave' in col or 'raman' in col or 'shift' in col), None)
    intensity_col = next((col for col in df.columns if 'intensity' in col or 'counts' in col), None)
    
    # If key columns are missing, assume the first two columns
    if not wavenumber_col and len(df.columns) >= 1:
        wavenumber_col = df.columns[0]
    
    if not intensity_col and len(df.columns) >= 2:
        intensity_col = df.columns[1]
    
    # Ensure we're using consistent names
    df = df.rename(columns={wavenumber_col: 'wavenumber', intensity_col: 'intensity'})
    wavenumber_col = 'wavenumber'
    intensity_col = 'intensity'
    
    # Perform peak finding and analysis
    try:
        # Find peaks
        peak_indices, peak_properties = find_peaks(df[intensity_col], height=0.2*df[intensity_col].max(),
                                                 distance=20, prominence=0.1*df[intensity_col].max())
        
        peak_positions = df[wavenumber_col].iloc[peak_indices]
        peak_heights = df[intensity_col].iloc[peak_indices]
        
        # Analyze peak properties
        insights.append(html.H5("Raman Peak Analysis"))
        insights.append(html.P(f"Number of significant peaks detected: {len(peak_positions)}"))
        
        # Add analysis data
        analysis_data["num_peaks"] = len(peak_positions)
        analysis_data["peak_positions"] = peak_positions.tolist()
        analysis_data["peak_intensities"] = peak_heights.tolist()
        
        # List the most intense peaks
        top_peaks = sorted(zip(peak_positions, peak_heights), key=lambda x: x[1], reverse=True)[:5]
        
        insights.append(html.P("Top peaks detected:"))
        peaks_list = []
        for i, (pos, height) in enumerate(top_peaks):
            peaks_list.append(html.Li(f"Peak at {pos:.1f} cm⁻¹ (Intensity: {height:.0f})"))
        
        insights.append(html.Ul(peaks_list))
        
        # Carbon material analysis (D and G bands)
        if df[wavenumber_col].min() < 1400 and df[wavenumber_col].max() > 1600:
            # Check if there are peaks in D and G band regions
            d_band_peaks = peak_positions[(peak_positions >= 1320) & (peak_positions <= 1380)]
            g_band_peaks = peak_positions[(peak_positions >= 1570) & (peak_positions <= 1630)]
            
            if len(d_band_peaks) > 0 and len(g_band_peaks) > 0:
                d_band = d_band_peaks.iloc[0]
                g_band = g_band_peaks.iloc[0]
                
                # Get the intensities
                d_intensity = df[df[wavenumber_col] == d_band][intensity_col].iloc[0]
                g_intensity = df[df[wavenumber_col] == g_band][intensity_col].iloc[0]
                
                # Calculate I(D)/I(G) ratio
                id_ig_ratio = d_intensity / g_intensity
                
                insights.append(html.H5("Carbon Material Analysis"))
                insights.append(html.P(f"D band position: {d_band:.1f} cm⁻¹"))
                insights.append(html.P(f"G band position: {g_band:.1f} cm⁻¹"))
                insights.append(html.P(f"I(D)/I(G) ratio: {id_ig_ratio:.2f}"))
                
                # Add analysis data
                analysis_data["d_band"] = float(d_band)
                analysis_data["g_band"] = float(g_band)
                analysis_data["id_ig_ratio"] = float(id_ig_ratio)
                
                # Provide context about the I(D)/I(G) ratio
                if id_ig_ratio < 0.5:
                    insights.append(html.P("The low I(D)/I(G) ratio indicates high graphitic ordering and few defects.", 
                                         className="text-success"))
                elif id_ig_ratio < 1.0:
                    insights.append(html.P("The moderate I(D)/I(G) ratio suggests a balance of graphitic domains and defects.", 
                                         className="text-info"))
                else:
                    insights.append(html.P("The high I(D)/I(G) ratio indicates a high level of defects or amorphous carbon.", 
                                         className="text-warning"))
    
    except Exception as e:
        print(f"Error in Raman peak analysis: {e}")
        insights.append(html.P("Unable to perform detailed peak analysis on the provided data.", 
                             className="text-danger"))
    
    # If no insights were generated
    if not insights:
        insights.append(html.P("No detailed analysis could be performed with the available data."))
    
    return html.Div(insights), analysis_data


def analyze_generic_data(df):
    """Analyze generic data types."""
    insights = []
    analysis_data = {}
    
    # Basic statistical analysis of numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        insights.append(html.P("No numeric data found for statistical analysis."))
        return html.Div(insights), analysis_data
    
    # Calculate basic statistics
    stats = df[numeric_cols].describe()
    
    # Add analysis data
    for col in numeric_cols:
        analysis_data[col] = {
            "mean": stats[col]["mean"],
            "std": stats[col]["std"],
            "min": stats[col]["min"],
            "max": stats[col]["max"],
        }
    
    # Generate insights
    insights.append(html.H5("Statistical Analysis"))
    
    for col in numeric_cols[:5]:  # Limit to first 5 columns
        col_insights = []
        col_insights.append(html.H6(f"Column: {col}"))
        col_insights.append(html.P(f"Mean: {stats[col]['mean']:.4g}"))
        col_insights.append(html.P(f"Std Dev: {stats[col]['std']:.4g}"))
        col_insights.append(html.P(f"Range: {stats[col]['min']:.4g} - {stats[col]['max']:.4g}"))
        
        # Check for outliers (values more than 3 std dev from mean)
        mean = stats[col]['mean']
        std = stats[col]['std']
        outliers = df[(df[col] < mean - 3*std) | (df[col] > mean + 3*std)][col]
        
        if len(outliers) > 0:
            col_insights.append(html.P(f"Outliers detected: {len(outliers)} values outside 3σ from mean", 
                                     className="text-warning"))
        
        insights.append(html.Div(col_insights, className="mb-4"))
    
    # Calculate correlations between numeric columns
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        
        # Find strong correlations (|r| > 0.7)
        strong_corrs = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                r = corr_matrix.iloc[i, j]
                if abs(r) > 0.7:
                    strong_corrs.append((numeric_cols[i], numeric_cols[j], r))
        
        if strong_corrs:
            corr_insights = [html.H5("Correlation Analysis")]
            corr_insights.append(html.P("Strong correlations detected:"))
            
            corr_items = []
            for col1, col2, r in strong_corrs:
                corr_type = "positive" if r > 0 else "negative"
                corr_items.append(html.Li(f"{col1} and {col2}: {r:.2f} ({corr_type})"))
            
            corr_insights.append(html.Ul(corr_items))
            insights.extend(corr_insights)
            
            # Add analysis data
            analysis_data["correlations"] = [
                {"col1": col1, "col2": col2, "r": float(r)} 
                for col1, col2, r in strong_corrs
            ]
    
    return html.Div(insights), analysis_data 