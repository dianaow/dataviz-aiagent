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
from app.utils.visualization import generate_llm_visualizations
from app.utils.analysis import analyze_data, generate_llm_analysis


def register_callbacks(app):
    """Register all callbacks for the application."""
    
    # Callback for file upload
    @app.callback(
        [
            Output("upload-status", "children"),
            Output("parsed-data", "data"),
            Output("sheet-selection-container", "style"),
            Output("sheet-selection-dropdown", "options"),
            Output("sheet-selection-dropdown", "value"),
        ],
        [Input("upload-data", "contents")],
        [State("upload-data", "filename")]
    )
    def update_upload_status(contents, filename):
        if contents is None:
            return (
                html.P("No file uploaded", className="text-muted"),
                None,
                {"display": "none"},
                [],
                None
            )
        
        try:
            # Parse the uploaded file
            result = parse_contents(contents, filename)
            
            if result is None:
                return (
                    html.P("Error processing file: Could not parse the file contents", className="text-danger"),
                    None,
                    {"display": "none"},
                    [],
                    None
                )
            
            # Handle multiple data tabs
            if isinstance(result, dict):
                # Convert JSON strings back to DataFrames for processing
                dataframes = {sheet_name: pd.read_json(json_str, orient='split') 
                            for sheet_name, json_str in result.items()}
                
                # Create dropdown options for sheet selection
                options = [{"label": sheet_name, "value": sheet_name} for sheet_name in dataframes.keys()]
                
                return (
                    html.P(f"File uploaded: {filename}. Please select a sheet to process.", className="text-success"),
                    result,
                    {"display": "block"},
                    options,
                    None
                )
            else:
                # Single DataFrame case (already in JSON format)
                return (
                    html.P(f"File uploaded: {filename}", className="text-success"),
                    result,
                    {"display": "none"},
                    [],
                    None
                )
                
        except Exception as e:
            return (
                html.P(f"Error processing file: {str(e)}", className="text-danger"),
                None,
                {"display": "none"},
                [],
                None
            )

    # Combined callback for sheet selection and data classification
    @app.callback(
        [
            Output("processed-data", "data", allow_duplicate=True),
            Output("sheet-selection-container", "style", allow_duplicate=True),
            Output("data-classification-output", "children"),
            Output("chat-history", "data"),
            Output("data-type", "data"),
            Output("column-matches", "data")
        ],
        [Input("process-button", "n_clicks")],
        [
            State("sheet-selection-dropdown", "value"),
            State("parsed-data", "data"),
            State("upload-data", "filename"),
            State("data-type-hint", "value")
        ],
        prevent_initial_call=True
    )
    def process_and_classify_data(n_clicks, selected_sheet, result, filename, data_type_hint):
        if n_clicks is None:
            return None, {"display": "none"}, None, [], None, None
        
        try:
            # Handle sheet selection
            if isinstance(result, dict):
                if selected_sheet is None:
                    return (
                        None,
                        {"display": "block"},
                        html.P("Please select a sheet to process", className="text-warning"),
                        [],
                        None,
                        None
                    )
                
                if selected_sheet not in result:
                    return (
                        None,
                        {"display": "none"},
                        html.P("Error: Selected sheet not found", className="text-danger"),
                        [],
                        None,
                        None
                    )
                
                # Get the selected sheet's data (already in JSON format)
                json_data = result[selected_sheet]
            else:
                # Single DataFrame case (already in JSON format)
                json_data = result

            # Convert JSON to DataFrame for processing
            df = pd.read_json(json_data, orient='split')

            # Classify the data
            data_type, confidence, column_matches, needs_confirmation = classify_data(df, data_type_hint, filename)
            
            # Initialize chat history
            current_time = pd.Timestamp.now().strftime("%H:%M:%S")
            chat_history = [{
                "role": "assistant",
                "content": f"I've analyzed your data and classified it as: {data_type} (Confidence: {confidence:.0%})",
                "timestamp": current_time
            }]
            
            # Create classification output
            classification_output = create_column_confirmation_dialog(
                column_matches,
                data_type,
                chat_history
            )
            
            return (
                json_data,  # Return the original JSON data
                {"display": "none"},
                classification_output,
                chat_history,
                data_type,
                column_matches
            )
            
        except Exception as e:
            print(f"Error processing and classifying data: {str(e)}")
            return (
                None,
                {"display": "none"},
                html.P(f"Error: {str(e)}", className="text-danger"),
                [],
                None,
                None
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
            State("chat-history", "data")
        ],
        prevent_initial_call=True
    )
    def handle_user_feedback(n_clicks, feedback, data_type, column_matches, chat_history):
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
            chat_history
        )
        
        return dialog, chat_history, result["updated_data_type"], result["updated_matches"]

    # Callback for handling classification confirmation
    @app.callback(
        [
            Output("plot-type-dropdown", "options"),
            Output("plot-type-dropdown", "disabled"),
            Output("plot-type-dropdown", "value"),
        ],
        [Input("confirm-classification", "n_clicks")],
        [State("data-type", "data")]
    )
    def confirm_output(n_clicks, data_type):
        if n_clicks is None:
            return dash.no_update
        
        options = get_plot_options(data_type)
        default_value = options[0]["value"] if options else None
        
        return options, False, default_value

    # Callback for visualization generation
    @app.callback(
        Output("visualization-container", "children"),
        [Input("confirm-classification", "n_clicks")],
        [
            State("processed-data", "data"), 
            State("data-type", "data"), 
            State("column-matches", "data")
        ],
        prevent_initial_call=True
    )
    def update_visualizations(n_clicks, json_data, data_type, column_matches):
        """Update visualizations based on data."""
        if n_clicks is None:
            return dash.no_update

        if not json_data or not data_type:
            return [html.P("Process data to see analysis results", className="text-muted")]
        
        try:
            # Convert JSON data to DataFrame
            df = pd.read_json(json_data, orient='split')
            
            # Rename columns based on matches
            for match in column_matches.get("required", []):
                if "old_column" in match and "new_column" in match:
                    df = df.rename(columns={match["old_column"]: match["new_column"]})
            
            # Log DataFrame info for debugging
            print(f"DataFrame columns: {df.columns.tolist()}")
            print(f"DataFrame shape: {df.shape}")
            
            # Generate all visualizations using LLM
            figures = generate_llm_visualizations(df, data_type)
            
            # Create a container for all figures with fixed height and scrolling
            visualization_container = html.Div([
                dcc.Graph(
                    figure=fig,
                    style={'height': '700px', 'margin-bottom': '20px'}
                ) for fig in figures
            ], style={
                'height': '700px',  # Fixed height for the container
                'overflow-y': 'auto',  # Enable vertical scrolling
                'padding': '20px'
            })
            
            return visualization_container
            
        except Exception as e:
            print(f"Error in visualization generation: {e}")
            return []

    @app.callback(
        [Output("analysis-results", "children")],
        [Input("confirm-classification", "n_clicks")],
        [
            State("processed-data", "data"), 
            State("data-type", "data")
        ],
    )
    def update_analysis_results(n_clicks, json_data, data_type):
        if n_clicks is None:
            return dash.no_update

        if not json_data or not data_type:
            return [html.P("Process data to see analysis results", className="text-muted")]
        
        try:
            # Convert JSON data to DataFrame
            df = pd.read_json(json_data, orient='split')
            
            # Generate analysis using LLM
            insights_layout, analysis_data = generate_llm_analysis(df, data_type)
            
            # Create a layout that shows insights and analysis data
            analysis_layout = html.Div([
                insights_layout,
                
                # Add detailed analysis data if available
                html.Div([
                    html.H5("Detailed Analysis Data", className="mt-4 mb-3"),
                    html.Div([
                        html.Div([
                            html.H6("Key Metrics", className="mb-3"),
                            html.Div([
                                html.Div([
                                    html.Strong(f"{metric}: "),
                                    str(value)
                                ]) for metric, value in analysis_data.get("key_metrics", {}).items()
                            ])
                        ], className="card mb-3 p-3"),
                        html.Div([
                            html.H6("Trends", className="mb-3"),
                            html.Div([
                                html.Div([
                                    html.Strong(trend["name"]), html.Br(),
                                    html.P(trend["description"], className="mb-1"),
                                    html.Small(f"Direction: {trend['direction']}, Magnitude: {trend['magnitude']}", 
                                             className="text-muted")
                                ], className="mb-3")
                                for trend in analysis_data.get("trends", [])
                            ])
                        ], className="card p-3")
                    ])
                ]) if analysis_data else None
            ])
            
            return [analysis_layout]
            
        except Exception as e:
            print(f"Error in analysis generation: {e}")
            return [html.P("Error generating analysis results", className="text-danger")]

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