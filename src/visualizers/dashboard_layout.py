"""Dashboard layout configuration."""

from dash import html, dcc

def create_layout(app):
    """Create the main dashboard layout."""
    return html.Div(
        [
            # Main container
            html.Div(
                [
                    # Top row - Title and key metrics
                    html.Div(
                        [
                            html.H1(
                                "Quantum Consciousness Mapper",
                                style={
                                    'textAlign': 'center',
                                    'margin': '20px 0',
                                    'color': '#2c3e50',
                                    'fontFamily': '"Inter", sans-serif'
                                }
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H3("Risk Level"),
                                            dcc.Graph(id='risk-gauge')
                                        ],
                                        style={'flex': '1', 'minWidth': '250px'}
                                    ),
                                    html.Div(
                                        [
                                            html.H3("System Status"),
                                            html.Div(
                                                id='system-status',
                                                style={'padding': '10px'}
                                            )
                                        ],
                                        style={'flex': '1', 'minWidth': '250px'}
                                    ),
                                    html.Div(
                                        [
                                            html.H3("Active Alerts"),
                                            html.Div(
                                                id='active-alerts',
                                                style={'padding': '10px'}
                                            )
                                        ],
                                        style={'flex': '1', 'minWidth': '250px'}
                                    )
                                ],
                                style={
                                    'display': 'flex',
                                    'flexWrap': 'wrap',
                                    'gap': '20px',
                                    'marginBottom': '20px',
                                    'justifyContent': 'space-between'
                                }
                            )
                        ]
                    ),

                    # Middle section - Patterns and Analysis
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("Neural Pattern Space"),
                                    dcc.Graph(id='pattern-space')
                                ],
                                style={'flex': '2', 'minWidth': '500px'}
                            ),
                            html.Div(
                                [
                                    html.H3("Quantum State Analysis"),
                                    dcc.Graph(id='quantum-state-analysis')
                                ],
                                style={'flex': '1', 'minWidth': '300px'}
                            )
                        ],
                        style={
                            'display': 'flex',
                            'flexWrap': 'wrap',
                            'gap': '20px',
                            'marginBottom': '20px'
                        }
                    ),

                    # Bottom section - Metrics and Stats
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("System Metrics"),
                                    dcc.Graph(id='metrics-timeline')
                                ],
                                style={'flex': '2', 'minWidth': '500px'}
                            ),
                            html.Div(
                                [
                                    html.H3("Quantum Statistics"),
                                    dcc.Graph(id='quantum-transitions'),
                                    html.Div(
                                        id='quantum-stats',
                                        style={'marginTop': '20px'}
                                    )
                                ],
                                style={'flex': '1', 'minWidth': '300px'}
                            )
                        ],
                        style={
                            'display': 'flex',
                            'flexWrap': 'wrap',
                            'gap': '20px',
                            'marginBottom': '20px'
                        }
                    ),

                    # Update interval
                    dcc.Interval(
                        id='interval-component',
                        interval=500,  # 500ms refresh
                        n_intervals=0
                    )
                ],
                style={
                    'padding': '20px',
                    'backgroundColor': '#f8f9fa'
                }
            )
        ],
        style={
            'minHeight': '100vh',
            'backgroundColor': '#f8f9fa',
            'fontFamily': '"Inter", sans-serif'
        }
    )