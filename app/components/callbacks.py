import json
import base64
import io
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash
import dash_table
import dash_html_components as html

from app.utils.data_processor import process_data, parse_contents
from app.utils.llm_classifier import classify_data, create_column_confirmation_dialog, process_user_feedback
from app.utils.visualization import generate_visualizations
from app.utils.analysis import analyze_data


def register_callbacks(app):
    """Register all callbacks for the application."""
    
    # Callback for file upload
    @app.callback(
        [
            Output("upload-status", "children"),
            Output("processed-data", "data"),
        ],
        [Input("upload-data", "contents")],
        [State("upload-data", "filename")]
    )
    def update_upload_status(contents, filename):
        if contents is None:
            return html.P("No file uploaded", className="text-muted"), None
        
        try:
            # Parse the uploaded file
            df = parse_contents(contents, filename)
            if df is None:
                return html.P("Error processing file: Could not parse the file contents", className="text-danger"), None
            return (
                html.P(f"File uploaded: {filename}", className="text-success"),
                df.to_json(date_format='iso', orient='split')
            )
        except Exception as e:
            return html.P(f"Error processing file: {str(e)}", className="text-danger"), None

    # Callback for data classification
    @app.callback(
        [
            Output("data-classification-output", "children"),
            Output("chat-history", "data"),
            Output("data-type", "data"),
            Output("data-summary", "children"),
            Output("column-matches", "data")
        ],
        [Input("process-button", "n_clicks")],
        [
            State("processed-data", "data"),
            State("data-type-hint", "value"),
            State("upload-data", "filename"),
        ]
    )
    def classify_and_summarize(n_clicks, json_data, data_type_hint, filename):
        if n_clicks is None or json_data is None:
            return (
                html.P("Upload and process data to see classification results", className="text-muted"),
                [],
                [],
                None,
                [],
            )
        
        # Parse the JSON data back to a DataFrame
        df = pd.read_json(json_data, orient='split')
        
        # Classify the data type using the LLM
        data_type, confidence, column_matches, needs_confirmation = classify_data(df, data_type_hint, filename)
        
        # Generate data summary
        data_summary = generate_data_summary(df, data_type)
        
        # Initialize chat history with the classification results
        current_time = pd.Timestamp.now().strftime("%H:%M:%S")
        chat_history = [{
            "role": "assistant",
            "content": f"I've analyzed your data and classified it as: {data_type} (Confidence: {confidence:.0%})",
            "timestamp": current_time
        }]
        
        # Create the chatbot-style dialog with data summary
        classification_output = create_column_confirmation_dialog(
            column_matches, 
            data_type, 
            chat_history,
            data_summary
        )
        print(f"Data summary: {data_summary}")
        return (
            classification_output,
            chat_history,
            data_type,
            data_summary,
            column_matches
        )
    
    # Callback for handling chatbot interactions between user and assistant
    @app.callback(
        [
            Output("data-classification-output", "children", allow_duplicate=True),
            Output("chat-history", "data", allow_duplicate=True),
            Output("data-type", "data", allow_duplicate=True),
            Output("column-matches", "data", allow_duplicate=True),
        ],
        [Input("send-feedback-button", "n_clicks")],
        [
            State("user-feedback-input", "value"),
            State("data-type", "data"),
            State("column-matches", "data"),
            State("chat-history", "data"),
            State("data-summary", "children"),
        ],
        prevent_initial_call=True
    )
    def handle_user_feedback(n_clicks, feedback, data_type, column_matches, chat_history, data_summary):
        if n_clicks is None or not feedback:
            return dash.no_update
        
        # Add user message to chat history
        current_time = pd.Timestamp.now().strftime("%H:%M:%S")
        chat_history.append({
            "role": "user",
            "content": feedback,
            "timestamp": current_time
        })
        
        # Process the feedback
        result = process_user_feedback(feedback, column_matches, data_type)
        
        # Add assistant's response to chat history
        current_time = pd.Timestamp.now().strftime("%H:%M:%S")
        chat_history.append({
            "role": "assistant",
            "content": result["explanation"],
            "timestamp": current_time
        })
        
        # Update the dialog with new chat history and data summary
        dialog = create_column_confirmation_dialog(
            result["updated_matches"], 
            data_type, 
            chat_history,
            data_summary
        )
        
        return dialog, chat_history, result["updated_data_type"], result["updated_matches"]

    # Callback for handling classification confirmation
    @app.callback(
        [
            Output("plot-type-dropdown", "options"),
            Output("plot-type-dropdown", "disabled"),
        ],
        [Input("confirm-classification", "n_clicks")],
        [State("data-type", "data")]
    )
    def confirm_output(n_clicks, data_type):
        if n_clicks is None:
            return dash.no_update
        
        return get_plot_options(data_type), False

    # Callback for plot type selection
    @app.callback(
        Output("primary-visualization", "figure"),
        [Input("plot-type-dropdown", "value")],
        [State("processed-data", "data"), State("data-type", "data"), State("column-matches", "data")]
    )
    def update_visualizations(plot_type, json_data, data_type, column_matches):
        print(f"Column matches: {column_matches}")
        print(f"Visualization callback triggered with plot_type: {plot_type}")
        print(f"Data type: {data_type}")
        print(f"JSON data available: {json_data is not None}")

        # Prevent callback execution on initial load
        if dash.ctx.triggered_id is None:
            return dash.no_update  # This prevents unnecessary updates

        if not plot_type or not json_data or not data_type:
            print("Missing required data for visualization")
            return dash.no_update  # Do not update the plot if required inputs are missing

        try:
            df = pd.read_json(json_data, orient='split')
            column_rename_map_required = {item['old_column']: item['new_column'] for item in column_matches['required']}
            column_rename_map_optional = {item['old_column']: item['new_column'] for item in column_matches['optional']}
            df.rename(columns=column_rename_map_required, inplace=True)
            df.rename(columns=column_rename_map_optional, inplace=True)
            print(df.columns)
            print(f"Column matches: {column_matches}")
            print(f"DataFrame shape: {df.shape}")
            print(f"DataFrame columns: {df.columns.tolist()}")
            
            # Get the primary and secondary figures
            primary_fig, secondary_fig = generate_visualizations(df, data_type, plot_type)
            print("Generated visualizations successfully")

            return primary_fig
            
        except Exception as e:
            print(f"Error generating visualizations: {str(e)}")
            return {}

    @app.callback(
        [Output("analysis-results", "children")],
        [Input("confirm-classification", "n_clicks")],
        [State("processed-data", "data"), State("data-type", "data")],
    )
    def update_analysis_results(n_clicks, json_data, data_type):
        if n_clicks is None:
            return dash.no_update

        if not json_data or not data_type:
            return [html.P("Process data to see analysis results", className="text-muted")]
        
        df = pd.read_json(json_data, orient='split')
        
        # Unpack both insights and analysis_data
        insights, analysis_data = analyze_data(df, data_type)
        
        # Create a layout that shows both insights and a detailed analysis data table
        analysis_layout = html.Div([
            insights,  # This will display the HTML insights from the function
            
            # Optional: Add a table to show the detailed analysis data
            html.H5("Detailed Analysis Data"),
            # dash_table.DataTable(
            #     id='analysis-data-table',
            #     columns=[{"name": str(k), "id": str(k)} for k in analysis_data.keys()],
            #     data=[analysis_data],
            #     style_table={'overflowX': 'auto'},
            #     style_cell={
            #         'textAlign': 'left',
            #         'padding': '5px',
            #         'backgroundColor': 'rgb(250, 250, 250)',
            #         'color': 'black'
            #     },
            #     style_header={
            #         'backgroundColor': 'rgb(230, 230, 230)',
            #         'fontWeight': 'bold'
            #     }
            # )
        ])
        
        return [analysis_layout]

    def generate_data_summary(df, data_type):
        """Generate a summary of the uploaded data based on its type."""
        summary_items = []
        
        # Basic data summary
        summary_items.append(html.P(f"Number of rows: {len(df)}"))
        summary_items.append(html.P(f"Number of columns: {len(df.columns)}"))
        
        # Add columns preview
        #summary_items.append(html.P("Columns:"))
        #summary_items.append(html.Ul([html.Li(col) for col in df.columns]))
        
        # Add type-specific summaries
        if data_type == "Battery Cycling Data":
            # For battery data, show cycle count, capacity range, etc.
            cycle_col = next((col for col in df.columns if 'cycle' in col.lower()), None)
            capacity_col = next((col for col in df.columns if 'capacity' in col.lower()), None)
            
            if cycle_col and cycle_col in df.columns:
                cycles = df[cycle_col].nunique()
                summary_items.append(html.P(f"Total cycles: {cycles}"))
            
            if capacity_col and capacity_col in df.columns:
                min_cap = df[capacity_col].min()
                max_cap = df[capacity_col].max()
                summary_items.append(html.P(f"Capacity range: {min_cap:.2f} - {max_cap:.2f}"))
        
        elif data_type == "X-ray Diffraction":
            # For XRD data, show 2-theta range, intensity range, etc.
            angle_col = next((col for col in df.columns if 'theta' in col.lower() or '2theta' in col.lower() 
                            or 'angle' in col.lower()), df.columns[0])
            intensity_col = next((col for col in df.columns if 'intensity' in col.lower() 
                                or 'counts' in col.lower()), df.columns[1])
            
            if angle_col in df.columns and intensity_col in df.columns:
                min_angle = df[angle_col].min()
                max_angle = df[angle_col].max()
                max_intensity = df[intensity_col].max()
                
                summary_items.append(html.P(f"Angle range: {min_angle:.2f}° - {max_angle:.2f}°"))
                summary_items.append(html.P(f"Max intensity: {max_intensity:.0f} counts"))
        
        elif data_type == "Raman Spectroscopy":
            # For Raman data, show wavenumber range, etc.
            wavenumber_col = next((col for col in df.columns if 'wave' in col.lower() 
                                  or 'raman' in col.lower() or 'shift' in col.lower()), df.columns[0])
            intensity_col = next((col for col in df.columns if 'intensity' in col.lower() 
                                or 'counts' in col.lower()), df.columns[1])
            
            if wavenumber_col in df.columns and intensity_col in df.columns:
                min_wave = df[wavenumber_col].min()
                max_wave = df[wavenumber_col].max()
                
                summary_items.append(html.P(f"Wavenumber range: {min_wave:.0f} - {max_wave:.0f} cm⁻¹"))
        
        return summary_items


def get_plot_options(data_type):
    """Get plot options based on data type."""
    if data_type == "Battery Cycling Data":
        return [
            {"label": "Capacity vs Cycle", "value": "capacity_cycle"},
            {"label": "Voltage vs Time", "value": "voltage_time"},
            {"label": "Battery Current vs. Time", "value": "current_time"},
            {"label": "Capacity Retention vs Cycle Number", "value": "retention_cycle"}
        ]
    elif data_type == "X-ray Diffraction":
        return [
            {"label": "Diffraction Pattern", "value": "diffraction_pattern"}
        ]
    elif data_type == "Raman Spectroscopy":
        return [
            {"label": "Raman Spectrum", "value": "raman_spectrum"}
        ]
    else:
        return [
            {"label": "Time Series", "value": "time_series"},
            {"label": "Scatter Plot", "value": "scatter_plot"},
            {"label": "Histogram", "value": "histogram"}
        ]