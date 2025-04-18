import dash_bootstrap_components as dbc
from dash import html, dcc

def create_layout():
    """Create the main application layout."""
    return dbc.Container([
        # Main Content Row
        dbc.Row([
            # Left Column: Data Input
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Data Input"),
                    dbc.CardBody([
                        # File Upload Component
                        html.Div([
                            html.H5("Upload Data File"),
                            dcc.Upload(
                                id='upload-data',
                                children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select Files')
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px 0'
                                },
                                multiple=False
                            ),
                            html.Div(id='upload-status'),
                            html.Div([
                            html.H4("Select Sheet", className="mb-3"),
                            dcc.Dropdown(id='sheet-selection-dropdown'),
                        ], id='sheet-selection-container', style={'display': 'none'}, className="card p-3 mb-4"),                           
                        ], className="mb-3"),
                        
                        # Data Type Hint Input
                        html.Div([
                            html.H5("Data Type Hint (Optional)"),
                            dbc.Input(
                                id="data-type-hint",
                                placeholder="E.g., 'Battery cycling data' or 'XRD'",
                                type="text",
                                className="mb-2"
                            ),
                            dbc.FormText("Provide a hint about the type of data to help with classification"),
                        ], className="mb-4"),
                        
                        # Process Data Button
                        dbc.Button("Process Data", id="process-button", color="primary", className="w-100"),
                    ])
                ], className="mb-4"),
            ], width=12, lg=2),  # First column (Data Input)

            # Middle Column: Data Classification
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Data Classification"),
                    dbc.CardBody([
                        html.Div(id="data-classification-output", children=[
                            html.P("Upload data to see classification results", className="text-muted")
                        ]),
                        dcc.Store(id='chat-history', data=[])
                    ])
                ])
            ], width=12, lg=3),  # Second column (Data Classification)

            # Right Column: Results
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Results"),
                    dbc.CardBody([
                        # dbc.Row([
                        #     # Plot Type Dropdown
                        #     dbc.Col([
                        #         html.Label("Plot Type"),
                        #         dcc.Dropdown(
                        #             id="plot-type-dropdown",
                        #             options=[],  # Will be populated dynamically
                        #             placeholder="Select a plot type",
                        #             disabled=True,
                        #             style={"width": "100%"},  
                        #             className="mb-4"
                        #         )
                        #     ], width=4),
                        # ]),
                        
                        # Visualization Tabs
                        dbc.Tabs([
                            dbc.Tab(
                                dcc.Loading(
                                    html.Div(id="visualization-container", style={"height": "700px"}),
                                    type="circle"
                                ),
                                label="Visualizations"
                            ),
                            # dbc.Tab(
                            #     dcc.Loading(
                            #         dcc.Graph(id="primary-visualization", style={"height": "700px"}),
                            #         type="circle"
                            #     ),
                            #     label="Primary Plot"
                            # ),
                            dbc.Tab(
                                dcc.Loading(
                                    html.Div(id="analysis-results", style={"height": "700px", "overflow": "scroll"}),
                                    type="circle"
                                ),
                                label="Analysis"
                            ),
                            dbc.Tab(
                                dcc.Loading(
                                    html.Div(id="comparison-results", style={"height": "700px", "overflow": "scroll"}),
                                    type="circle"
                                ),
                                label="Comparison"
                            )
                        ])
                    ])
                ])
            ], width=12, lg=7)  # Third column (Visualizations)
        ]),

        # Store components for intermediate data
        dcc.Store(id="processed-data"),
        dcc.Store(id="data-type"),
        dcc.Store(id="analysis-output"),
        dcc.Store(id="column-matches"),
    ], 
    fluid=True,
    className="min-vh-90 d-flex flex-column"
    )
