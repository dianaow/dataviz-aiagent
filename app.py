import os
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash.dependencies import Input, Output, State
import flask
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import components
from app.components.layout import create_layout
from app.utils.data_processor import process_data
from app.utils.llm_classifier import classify_data
from app.utils.visualization import generate_visualizations

# Initialize the app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server
app.title = "Advanced Materials Data Analyzer"

# Set the layout
app.layout = create_layout()

# Include callbacks
from app.components.callbacks import register_callbacks
register_callbacks(app)

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    app.run_server(debug=debug, port=port)
