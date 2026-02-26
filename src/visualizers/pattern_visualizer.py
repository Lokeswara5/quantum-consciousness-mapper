import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta
from ..core.quantum_state_detector import QuantumStateMetrics

class PatternVisualizer:
    """Visualize AI system patterns and behaviors"""

    def __init__(self):
        self.color_scale = px.colors.qualitative.Set3
        self.time_window = timedelta(minutes=30)  # Default time window
        self.state_colors = {
            "GHZ": "#FF4B4B",  # Red
            "W": "#4B9EFF",    # Blue
            "other": "#808080"  # Gray
        }

    def create_3d_pattern_plot(self,
                             coordinates: np.ndarray,
                             pattern_labels: Optional[List[int]] = None,
                             quantum_metrics: Optional[QuantumStateMetrics] = None,
                             title: str = "Neural Pattern Space") -> go.Figure:
        """Create 3D visualization of neural patterns"""
        if coordinates.shape[1] != 3:
            raise ValueError("Coordinates must be in 3D space")

        # Create figure
        fig = go.Figure()

        # Add points with quantum state coloring if available
        if quantum_metrics is not None:
            # Color points by quantum state type
            color = self.state_colors[quantum_metrics.state_type]
            # Adjust opacity based on state confidence
            opacity = max(0.3, quantum_metrics.confidence)

            fig.add_trace(go.Scatter3d(
                x=coordinates[:, 0],
                y=coordinates[:, 1],
                z=coordinates[:, 2],
                mode='markers',
                marker=dict(
                    size=6,
                    color=color,
                    opacity=opacity,
                    symbol='circle',
                    line=dict(color='darkgray', width=1)
                ),
                name=f'{quantum_metrics.state_type} State'
            ))

            # Add confidence sphere
            if quantum_metrics.state_type in ["GHZ", "W"]:
                center = coordinates.mean(axis=0)
                radius = quantum_metrics.coherence_score
                sphere_points = self._create_sphere(center, radius)

                fig.add_trace(go.Scatter3d(
                    x=sphere_points[0],
                    y=sphere_points[1],
                    z=sphere_points[2],
                    mode='lines',
                    line=dict(color=color, width=1),
                    opacity=0.2,
                    name=f'Coherence Sphere ({quantum_metrics.coherence_score:.2f})'
                ))

        elif pattern_labels is not None:
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

    def _create_sphere(self, center: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create sphere points for visualization"""
        phi = np.linspace(0, np.pi, 20)
        theta = np.linspace(0, 2 * np.pi, 40)
        phi, theta = np.meshgrid(phi, theta)

        x = center[0] + radius * np.sin(phi) * np.cos(theta)
        y = center[1] + radius * np.sin(phi) * np.sin(theta)
        z = center[2] + radius * np.cos(phi)

        return x, y, z

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

    def create_quantum_state_plot(self,
                                metrics: QuantumStateMetrics,
                                title: str = "Quantum State Analysis") -> go.Figure:
        """Create visualization of quantum state properties"""
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "polar", "rowspan": 2}, {"type": "indicator"}],
                  [None, {"type": "bar"}]],
            subplot_titles=("Phase Space", "Coherence", "Entanglement Measures")
        )

        # Add phase space plot (polar plot)
        theta = np.linspace(0, 2*np.pi, 100)
        radius = metrics.coherence_score * np.ones_like(theta)
        fig.add_trace(
            go.Scatterpolar(
                r=radius,
                theta=theta,
                fill='toself',
                name=metrics.state_type,
                line_color=self.state_colors[metrics.state_type]
            ),
            row=1, col=1
        )

        # Add coherence gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics.coherence_score * 100,
                domain=dict(x=[0.5, 1], y=[0.5, 1]),
                title=dict(text="Coherence"),
                gauge=dict(
                    axis=dict(range=[0, 100]),
                    bar=dict(color=self.state_colors[metrics.state_type]),
                    steps=[
                        dict(range=[0, 30], color="red"),
                        dict(range=[30, 70], color="yellow"),
                        dict(range=[70, 100], color="green")
                    ],
                    threshold=dict(
                        line=dict(color="red", width=4),
                        thickness=0.75,
                        value=80
                    )
                )
            ),
            row=1, col=2
        )

        # Add entanglement measures
        measures = metrics.entanglement_measures
        fig.add_trace(
            go.Bar(
                x=list(measures.keys()),
                y=list(measures.values()),
                marker_color=self.state_colors[metrics.state_type]
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            height=800,
            width=1000,
            title_text=title,
            showlegend=False
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
                                    quantum_metrics: Optional[List[QuantumStateMetrics]] = None,
                                    title: str = "Pattern Evolution") -> go.Figure:
        """Visualize pattern evolution over time"""
        fig = go.Figure()

        for i, trajectory in enumerate(pattern_trajectories):
            color = self.color_scale[i % len(self.color_scale)]
            opacity = 0.8
            name = f'Pattern {i+1}'

            # Use quantum state coloring if available
            if quantum_metrics and i < len(quantum_metrics):
                metrics = quantum_metrics[i]
                color = self.state_colors[metrics.state_type]
                opacity = max(0.3, metrics.confidence)
                name = f'{metrics.state_type} State {i+1}'

            # Plot 3D trajectory
            fig.add_trace(go.Scatter3d(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                z=trajectory[:, 2],
                mode='lines+markers',
                name=name,
                line=dict(
                    color=color,
                    width=2
                ),
                marker=dict(
                    size=4,
                    color=np.arange(len(trajectory)),
                    colorscale='Viridis',
                    opacity=opacity,
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

    def create_quantum_transition_matrix(self,
                                     transition_matrix: np.ndarray,
                                     title: str = "Quantum State Transitions") -> go.Figure:
        """Create visualization of quantum state transition probabilities"""
        states = ["GHZ", "W", "other"]

        fig = go.Figure(data=go.Heatmap(
            z=transition_matrix,
            x=states,
            y=states,
            colorscale='Viridis',
            text=np.round(transition_matrix, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))

        fig.update_layout(
            title=title,
            xaxis_title="To State",
            yaxis_title="From State",
            height=500,
            width=600
        )

        return fig