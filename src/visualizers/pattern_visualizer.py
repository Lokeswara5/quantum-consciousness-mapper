import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta

class PatternVisualizer:
    """Visualize AI system patterns and behaviors"""

    def __init__(self):
        self.color_scale = px.colors.qualitative.Set3
        self.time_window = timedelta(minutes=30)  # Default time window

    def create_3d_pattern_plot(self,
                             coordinates: np.ndarray,
                             pattern_labels: Optional[List[int]] = None,
                             title: str = "Neural Pattern Space") -> go.Figure:
        """Create 3D visualization of neural patterns"""
        if coordinates.shape[1] != 3:
            raise ValueError("Coordinates must be in 3D space")

        # Create figure
        fig = go.Figure()

        # Add points
        if pattern_labels is not None:
            unique_labels = np.unique(pattern_labels)
            for i, label in enumerate(unique_labels):
                mask = pattern_labels == label
                fig.add_trace(go.Scatter3d(
                    x=coordinates[mask, 0],
                    y=coordinates[mask, 1],
                    z=coordinates[mask, 2],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=self.color_scale[i % len(self.color_scale)],
                        opacity=0.8
                    ),
                    name=f'Pattern {label}'
                ))
        else:
            fig.add_trace(go.Scatter3d(
                x=coordinates[:, 0],
                y=coordinates[:, 1],
                z=coordinates[:, 2],
                mode='markers',
                marker=dict(
                    size=6,
                    color=coordinates[:, 2],  # Color by z-coordinate
                    colorscale='Viridis',
                    opacity=0.8
                )
            ))

        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                zaxis_title="Dimension 3"
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )

        return fig

    def create_stability_heatmap(self,
                               stability_matrix: np.ndarray,
                               layer_names: List[str],
                               title: str = "Layer Stability Heatmap") -> go.Figure:
        """Create heatmap of layer stability metrics"""
        fig = go.Figure(data=go.Heatmap(
            z=stability_matrix,
            x=layer_names,
            y=layer_names,
            colorscale='RdYlBu',
            text=np.round(stability_matrix, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Layer",
            yaxis_title="Layer",
            height=600,
            width=800
        )

        return fig

    def create_time_series_dashboard(self,
                                   metrics: Dict[str, List[float]],
                                   timestamps: List[datetime],
                                   title: str = "System Metrics Over Time") -> go.Figure:
        """Create multi-metric time series dashboard"""
        n_metrics = len(metrics)
        fig = make_subplots(
            rows=n_metrics,
            cols=1,
            subplot_titles=list(metrics.keys()),
            vertical_spacing=0.05
        )

        for i, (metric_name, values) in enumerate(metrics.items(), 1):
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=values,
                    name=metric_name,
                    mode='lines+markers',
                    line=dict(
                        color=self.color_scale[i % len(self.color_scale)]
                    )
                ),
                row=i,
                col=1
            )

            # Add threshold lines if needed
            if 'stability' in metric_name.lower():
                fig.add_hline(
                    y=0.7,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Threshold",
                    row=i,
                    col=1
                )

        fig.update_layout(
            height=300 * n_metrics,
            width=1000,
            showlegend=False,
            title_text=title
        )

        return fig

    def create_pattern_evolution_plot(self,
                                    pattern_trajectories: List[np.ndarray],
                                    timestamps: List[datetime],
                                    title: str = "Pattern Evolution") -> go.Figure:
        """Visualize pattern evolution over time"""
        fig = go.Figure()

        for i, trajectory in enumerate(pattern_trajectories):
            # Plot 3D trajectory
            fig.add_trace(go.Scatter3d(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                z=trajectory[:, 2],
                mode='lines+markers',
                name=f'Pattern {i+1}',
                line=dict(
                    color=self.color_scale[i % len(self.color_scale)],
                    width=2
                ),
                marker=dict(
                    size=4,
                    color=np.arange(len(trajectory)),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title='Time'
                    )
                )
            ))

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                zaxis_title="Dimension 3"
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )

        return fig

    def create_risk_gauge(self,
                         risk_level: float,
                         title: str = "System Risk Level") -> go.Figure:
        """Create a gauge chart for risk level visualization"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_level * 100,
            domain=dict(x=[0, 1], y=[0, 1]),
            title=dict(text=title),
            gauge=dict(
                axis=dict(range=[0, 100]),
                bar=dict(color="darkblue"),
                bgcolor="white",
                borderwidth=2,
                bordercolor="gray",
                steps=[
                    dict(range=[0, 30], color="green"),
                    dict(range=[30, 70], color="yellow"),
                    dict(range=[70, 90], color="orange"),
                    dict(range=[90, 100], color="red")
                ],
                threshold=dict(
                    line=dict(color="red", width=4),
                    thickness=0.75,
                    value=70
                )
            )
        ))

        fig.update_layout(
            height=400,
            width=600
        )

        return fig