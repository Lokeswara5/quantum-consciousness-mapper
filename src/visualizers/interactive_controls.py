from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
from typing import Dict, List, Optional
import plotly.graph_objects as go
from datetime import datetime, timedelta

class InteractiveControls:
    """Interactive control panel for AI system analysis"""

    @staticmethod
    def create_control_panel() -> html.Div:
        """Create main control panel"""
        return html.Div([
            # Time range selector
            html.Div([
                html.H4("Time Range"),
                dcc.RangeSlider(
                    id='time-range-slider',
                    min=0,
                    max=100,
                    step=1,
                    value=[80, 100],
                    marks={0: 'Start', 100: 'Now'},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], className='control-section'),

            # Pattern filters
            html.Div([
                html.H4("Pattern Filters"),
                dcc.Checklist(
                    id='pattern-type-filter',
                    options=[
                        {'label': 'Stable Patterns', 'value': 'stable'},
                        {'label': 'Emergent Patterns', 'value': 'emergent'},
                        {'label': 'Critical Patterns', 'value': 'critical'}
                    ],
                    value=['stable', 'emergent'],
                    inline=True
                )
            ], className='control-section'),

            # Layer selector
            html.Div([
                html.H4("Layer Selection"),
                dcc.Dropdown(
                    id='layer-selector',
                    multi=True,
                    placeholder="Select layers to analyze..."
                )
            ], className='control-section'),

            # Metric thresholds
            html.Div([
                html.H4("Alert Thresholds"),
                html.Div([
                    html.Label("Stability Threshold"),
                    dcc.Slider(
                        id='stability-threshold',
                        min=0,
                        max=1,
                        step=0.05,
                        value=0.7,
                        marks={0: '0', 0.5: '0.5', 1: '1'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ]),
                html.Div([
                    html.Label("Complexity Threshold"),
                    dcc.Slider(
                        id='complexity-threshold',
                        min=0,
                        max=1,
                        step=0.05,
                        value=0.8,
                        marks={0: '0', 0.5: '0.5', 1: '1'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ])
            ], className='control-section'),

            # Analysis tools
            html.Div([
                html.H4("Analysis Tools"),
                dcc.Tabs([
                    dcc.Tab(label='Pattern Comparison', children=[
                        html.Div([
                            dcc.Dropdown(
                                id='pattern-comparison-selector',
                                multi=True,
                                placeholder="Select patterns to compare..."
                            ),
                            html.Button(
                                'Compare Patterns',
                                id='compare-patterns-button',
                                className='control-button'
                            )
                        ])
                    ]),
                    dcc.Tab(label='Time Series Analysis', children=[
                        html.Div([
                            dcc.Checklist(
                                id='metric-selector',
                                options=[
                                    {'label': 'Stability', 'value': 'stability'},
                                    {'label': 'Complexity', 'value': 'complexity'},
                                    {'label': 'Risk Level', 'value': 'risk'},
                                    {'label': 'Pattern Count', 'value': 'patterns'}
                                ],
                                value=['stability', 'risk'],
                                inline=True
                            ),
                            html.Button(
                                'Update Analysis',
                                id='update-analysis-button',
                                className='control-button'
                            )
                        ])
                    ]),
                    dcc.Tab(label='Pattern Evolution', children=[
                        html.Div([
                            dcc.Dropdown(
                                id='evolution-pattern-selector',
                                placeholder="Select pattern to track..."
                            ),
                            dcc.RadioItems(
                                id='evolution-view-type',
                                options=[
                                    {'label': '3D Trajectory', 'value': '3d'},
                                    {'label': 'Phase Space', 'value': 'phase'},
                                    {'label': 'State Space', 'value': 'state'}
                                ],
                                value='3d',
                                inline=True
                            )
                        ])
                    ])
                ])
            ], className='control-section'),

            # Export controls
            html.Div([
                html.H4("Export Options"),
                html.Div([
                    dcc.Checklist(
                        id='export-options',
                        options=[
                            {'label': 'Pattern Data', 'value': 'patterns'},
                            {'label': 'Metrics History', 'value': 'metrics'},
                            {'label': 'Alert History', 'value': 'alerts'}
                        ],
                        value=['patterns', 'metrics'],
                        inline=True
                    ),
                    html.Button(
                        'Export Data',
                        id='export-button',
                        className='control-button'
                    )
                ])
            ], className='control-section'),

            # Custom analysis
            html.Div([
                html.H4("Custom Analysis"),
                dcc.Textarea(
                    id='custom-analysis-query',
                    placeholder="Enter custom analysis parameters...",
                    style={'width': '100%', 'height': 100}
                ),
                html.Button(
                    'Run Analysis',
                    id='run-custom-analysis',
                    className='control-button'
                )
            ], className='control-section')
        ], id='control-panel', className='control-panel')

    @staticmethod
    def create_analysis_panel() -> html.Div:
        """Create analysis results panel"""
        return html.Div([
            # Pattern comparison results
            html.Div([
                html.H4("Pattern Comparison"),
                dcc.Graph(id='pattern-comparison-plot'),
                html.Div(id='pattern-comparison-stats')
            ], id='pattern-comparison-section', className='analysis-section'),

            # Time series analysis results
            html.Div([
                html.H4("Time Series Analysis"),
                dcc.Graph(id='time-series-plot'),
                html.Div(id='time-series-stats')
            ], id='time-series-section', className='analysis-section'),

            # Pattern evolution results
            html.Div([
                html.H4("Pattern Evolution"),
                dcc.Graph(id='evolution-plot'),
                html.Div(id='evolution-stats')
            ], id='evolution-section', className='analysis-section'),

            # Custom analysis results
            html.Div([
                html.H4("Custom Analysis Results"),
                dcc.Graph(id='custom-analysis-plot'),
                html.Div(id='custom-analysis-results')
            ], id='custom-analysis-section', className='analysis-section')
        ], id='analysis-panel', className='analysis-panel')

class AnalysisCallbacks:
    """Callback handlers for interactive analysis"""

    @staticmethod
    def register_callbacks(app, dashboard):
        """Register all interactive callbacks"""

        @app.callback(
            [Output('pattern-comparison-plot', 'figure'),
             Output('pattern-comparison-stats', 'children')],
            [Input('compare-patterns-button', 'n_clicks')],
            [State('pattern-comparison-selector', 'value')]
        )
        def update_pattern_comparison(n_clicks, selected_patterns):
            if not n_clicks or not selected_patterns:
                return {}, html.Div("Select patterns to compare")

            # Generate comparison visualization
            fig = go.Figure()
            stats = []

            for pattern_id in selected_patterns:
                pattern = dashboard.current_network_state.activation_patterns[pattern_id]
                # Add pattern trace
                fig.add_trace(go.Scatter3d(
                    x=pattern.activation_pattern[:, 0],
                    y=pattern.activation_pattern[:, 1],
                    z=pattern.activation_pattern[:, 2],
                    mode='markers',
                    name=f'Pattern {pattern_id}'
                ))
                # Add stats
                stats.append(html.P([
                    html.Strong(f"Pattern {pattern_id}: "),
                    f"Complexity: {pattern.complexity:.3f}, ",
                    f"Stability: {pattern.stability:.3f}"
                ]))

            return fig, html.Div(stats)

        @app.callback(
            [Output('time-series-plot', 'figure'),
             Output('time-series-stats', 'children')],
            [Input('update-analysis-button', 'n_clicks')],
            [State('metric-selector', 'value'),
             State('time-range-slider', 'value')]
        )
        def update_time_series(n_clicks, selected_metrics, time_range):
            if not n_clicks or not selected_metrics:
                return {}, html.Div("Select metrics to analyze")

            # Create time series plot
            fig = go.Figure()
            stats = []

            start_idx = int(len(dashboard.timestamps) * time_range[0] / 100)
            end_idx = int(len(dashboard.timestamps) * time_range[1] / 100)

            for metric in selected_metrics:
                data = dashboard.metrics_history[metric][start_idx:end_idx]
                fig.add_trace(go.Scatter(
                    x=dashboard.timestamps[start_idx:end_idx],
                    y=data,
                    name=metric
                ))
                # Add stats
                stats.append(html.P([
                    html.Strong(f"{metric}: "),
                    f"Mean: {np.mean(data):.3f}, ",
                    f"Std: {np.std(data):.3f}"
                ]))

            return fig, html.Div(stats)

        @app.callback(
            Output('evolution-plot', 'figure'),
            [Input('evolution-pattern-selector', 'value'),
             Input('evolution-view-type', 'value')]
        )
        def update_evolution_plot(pattern_id, view_type):
            if not pattern_id:
                return {}

            fig = go.Figure()
            if view_type == '3d':
                # Create 3D trajectory plot
                trajectory = np.array([
                    state.activation_patterns[pattern_id].activation_pattern
                    for state in dashboard.pattern_history
                ])
                fig = dashboard.visualizer.create_pattern_evolution_plot(
                    [trajectory],
                    dashboard.timestamps
                )
            elif view_type == 'phase':
                # Create phase space plot
                pattern = dashboard.current_network_state.activation_patterns[pattern_id]
                fig.add_trace(go.Scatter(
                    x=pattern.activation_pattern[:, 0],
                    y=pattern.activation_pattern[:, 1],
                    mode='markers+lines',
                    name='Phase Space'
                ))
            else:  # state space
                # Create state space plot
                pattern = dashboard.current_network_state.activation_patterns[pattern_id]
                fig.add_trace(go.Heatmap(
                    z=pattern.activation_pattern,
                    colorscale='Viridis',
                    name='State Space'
                ))

            return fig