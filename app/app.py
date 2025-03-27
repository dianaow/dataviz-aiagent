import dash
from dash import Dash
import dash_bootstrap_components as dbc
from app.components.layout import create_layout
from app.components.callbacks import register_callbacks

# Initialize the Dash app with Bootstrap and custom CSS
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        '/assets/custom.css'  # This will automatically load from the assets folder
    ],
    suppress_callback_exceptions=True
)

# Create the app layout
app.layout = create_layout()

# Register all callbacks
register_callbacks(app)

# Run the server
if __name__ == '__main__':
    app.run_server(debug=True) 
    