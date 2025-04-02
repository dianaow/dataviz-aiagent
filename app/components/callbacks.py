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
                [],
            )
        
        # Parse the JSON data back to a DataFrame
        df = pd.read_json(json_data, orient='split')
        
        # Classify the data type using the LLM
        data_type, confidence, column_matches, needs_confirmation = classify_data(df, data_type_hint, filename)
         
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
            chat_history
        )

        return (
            classification_output,
            chat_history,
            data_type,
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