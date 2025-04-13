import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import datetime

class VisualizationHelper:
    """
    Handles creation of interactive visualizations for portfolio performance
    and asset allocation.
    """
    
    def __init__(self):
        """Initialize the visualization helper with default settings"""
        self.color_map = {
            "My Portfolio": "#1f77b4",  # Blue
            "S&P 500": "#ff7f0e",       # Orange
            "Nasdaq 100": "#2ca02c",    # Green
            "Euro Stoxx 50": "#d62728", # Red
            
            # Asset class colors
            "Money Market": "#aec7e8",
            "Bond": "#ffbb78",
            "Stock": "#98df8a",
            "ETF": "#ff9896",
            "REIT": "#c5b0d5",
            "Commodity": "#c49c94",
            "Cryptocurrency": "#f7b6d2",
            "Alternative": "#c7c7c7",
            "Mutual Fund": "#dbdb8d",
            "Index": "#9edae5",
            
            # Sector colors
            "Technology": "#1f77b4",
            "Healthcare": "#ff7f0e",
            "Financial": "#2ca02c",
            "Consumer Cyclical": "#d62728",
            "Consumer Defensive": "#9467bd",
            "Industrials": "#8c564b",
            "Basic Materials": "#e377c2",
            "Energy": "#7f7f7f",
            "Utilities": "#bcbd22",
            "Real Estate": "#17becf",
            "Communication Services": "#9edae5"
        }
        
        # Default chart settings
        self.default_height = 500
        self.default_template = "plotly_white"
    
    def create_performance_chart(
        self, 
        portfolio_data: pd.DataFrame, 
        benchmark_data: List[Dict], 
        period: str,
        show_absolute: bool = False,
        height: int = None
    ) -> go.Figure:
        """
        Create an interactive performance comparison chart with dual axis option.
        
        Args:
            portfolio_data: DataFrame with portfolio performance data
            benchmark_data: List of dictionaries with benchmark performance data
            period: Time period for the chart (e.g., "1d", "1wk", "1mo")
            show_absolute: Whether to show absolute values on a secondary axis
            height: Chart height in pixels
            
        Returns:
            Plotly figure object
        """
        if height is None:
            height = self.default_height
            
        # Prepare portfolio data
        if portfolio_data.empty:
            return self._create_empty_chart("No portfolio data available")
        
        # Ensure portfolio_data is valid and contains numeric values
        if not isinstance(portfolio_data, pd.Series) and not isinstance(portfolio_data, pd.DataFrame):
            return self._create_empty_chart("Invalid portfolio data format")
            
        # Convert to DataFrame if it's a Series
        if isinstance(portfolio_data, pd.Series):
            portfolio_norm = pd.DataFrame({
                'Date': portfolio_data.index,
                'Normalized': portfolio_data.values,
                'Index': "My Portfolio"
            })
        else:
            # If it's already a DataFrame, ensure it has the right format
            portfolio_norm = pd.DataFrame({
                'Date': portfolio_data.index,
                'Normalized': portfolio_data.iloc[:, 0].values if not portfolio_data.empty else [],
                'Index': "My Portfolio"
            })
        
        # Prepare benchmark data
        bench_dfs = []
        for b in benchmark_data:
            if b and "data" in b:
                bench_df = pd.DataFrame(b["data"])
                bench_df["Index"] = b["label"]
                bench_dfs.append(bench_df)
        
        # Combine data
        if bench_dfs:
            combined = pd.concat([portfolio_norm] + bench_dfs, ignore_index=True)
        else:
            combined = portfolio_norm
            
        # Ensure Date column contains datetime objects
        combined['Date'] = pd.to_datetime(combined['Date'], errors='coerce')
        
        # Drop rows with NaT dates
        combined = combined.dropna(subset=['Date'])
        
        # Sort by date
        combined = combined.sort_values("Date")
        
        # Ensure Normalized column contains numeric values
        combined['Normalized'] = pd.to_numeric(combined['Normalized'], errors='coerce')
        
        # Fill NaN values in Normalized column
        combined['Normalized'] = combined['Normalized'].fillna(0)
        
        if show_absolute:
            # Create a figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add normalized performance lines (left y-axis)
            for index_name in combined["Index"].unique():
                df_filtered = combined[combined["Index"] == index_name]
                fig.add_trace(
                    go.Scatter(
                        x=df_filtered["Date"],
                        y=df_filtered["Normalized"],
                        name=f"{index_name} (%)",
                        line=dict(color=self.color_map.get(index_name, "#1f77b4")),
                        hovertemplate="%{y:.2f}%<extra>%{fullData.name}</extra>"
                    ),
                    secondary_y=False
                )
            
            # Add absolute value line for portfolio only (right y-axis)
            if "Value" in portfolio_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=portfolio_data.index,
                        y=portfolio_data["Value"],
                        name="Portfolio Value ($)",
                        line=dict(color="#7f7f7f", dash="dash"),
                        hovertemplate="$%{y:,.2f}<extra>Portfolio Value</extra>"
                    ),
                    secondary_y=True
                )
            
            # Set axis titles
            fig.update_yaxes(title_text="Normalized Performance (%)", secondary_y=False)
            fig.update_yaxes(title_text="Portfolio Value ($)", secondary_y=True)
        else:
            # Create a simple line chart
            fig = px.line(
                combined,
                x="Date",
                y="Normalized",
                color="Index",
                height=height,
                template=self.default_template,
                color_discrete_map=self.color_map
            )
            
            # Update layout
            fig.update_layout(
                yaxis_title="Normalized Performance (%)"
            )
        
        # Common layout updates
        fig.update_layout(
            xaxis_title="Market Hours" if period == "1d" else "Date",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=height,
            template=self.default_template
        )
        
        # Format x-axis based on period
        if period == "1d":
            fig.update_xaxes(tickformat="%H:%M")
        elif period in ["1wk", "1mo", "3mo"]:
            fig.update_xaxes(tickformat="%b %d")
        else:
            fig.update_xaxes(tickformat="%b %Y")
            
        return fig
    
    def create_allocation_chart(
        self, 
        data: pd.DataFrame, 
        group_by: str, 
        title: str = None,
        height: int = None,
        show_values: bool = True
    ) -> go.Figure:
        """
        Create an interactive allocation pie chart.
        
        Args:
            data: DataFrame with allocation data
            group_by: Column to group by (e.g., "Asset Class", "Sector")
            title: Chart title
            height: Chart height in pixels
            show_values: Whether to show values in hover text
            
        Returns:
            Plotly figure object
        """
        if height is None:
            height = self.default_height // 2
            
        if data.empty:
            return self._create_empty_chart("No data available")
            
        # Ensure required columns exist
        if group_by not in data.columns:
            return self._create_empty_chart(f"Missing '{group_by}' column")
            
        if "Market Value" not in data.columns:
            return self._create_empty_chart("Missing 'Market Value' column")
            
        # Group data with error handling
        try:
            # Convert Market Value to numeric to avoid groupby errors
            data["Market Value"] = pd.to_numeric(data["Market Value"], errors="coerce")
            
            # Drop rows with NaN Market Value
            data = data.dropna(subset=["Market Value"])
            
            # Group data
            grouped = data.groupby(group_by, as_index=False)["Market Value"].sum()
            
            # Calculate percentages
            total = grouped["Market Value"].sum()
            if total <= 0:
                return self._create_empty_chart("Total market value is zero or negative")
                
            grouped["Percentage"] = grouped["Market Value"] / total * 100
        except Exception as e:
            import logging
            logging.warning(f"Error in create_allocation_chart: {str(e)}")
            return self._create_empty_chart(f"Error processing data: {str(e)}")
        
        # Create hover text
        if show_values:
            grouped["hover_text"] = grouped.apply(
                lambda x: f"{x[group_by]}: ${x['Market Value']:,.2f} ({x['Percentage']:.1f}%)", 
                axis=1
            )
        else:
            grouped["hover_text"] = grouped.apply(
                lambda x: f"{x[group_by]}: {x['Percentage']:.1f}%", 
                axis=1
            )
        
        # Create pie chart
        fig = px.pie(
            grouped,
            values="Market Value",
            names=group_by,
            title=title,
            height=height,
            template=self.default_template,
            color=group_by,
            color_discrete_map=self.color_map,
            hover_data=["hover_text"],
            custom_data=["hover_text"]
        )
        
        # Update layout
        fig.update_traces(
            textposition="inside",
            textinfo="percent",
            hovertemplate="%{customdata[0]}<extra></extra>",
            hole=0.4
        )
        
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        
        return fig
    
    def create_historical_allocation_chart(
        self, 
        data: pd.DataFrame, 
        group_by: str, 
        title: str = None,
        height: int = None
    ) -> go.Figure:
        """
        Create an interactive historical allocation area chart.
        
        Args:
            data: DataFrame with historical allocation data
            group_by: Column to group by (e.g., "Asset Class", "Sector")
            title: Chart title
            height: Chart height in pixels
            
        Returns:
            Plotly figure object
        """
        if height is None:
            height = self.default_height
            
        if data.empty:
            return self._create_empty_chart("No historical data available")
            
        # Create area chart
        fig = px.area(
            data,
            x="Date",
            y="Market Value",
            color=group_by,
            title=title,
            height=height,
            template=self.default_template,
            color_discrete_map=self.color_map
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Market Value ($)",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_performance_heatmap(
        self, 
        data: pd.DataFrame, 
        title: str = None,
        height: int = None
    ) -> go.Figure:
        """
        Create an interactive performance heatmap.
        
        Args:
            data: DataFrame with performance data
            title: Chart title
            height: Chart height in pixels
            
        Returns:
            Plotly figure object
        """
        if height is None:
            height = self.default_height
            
        if data.empty:
            return self._create_empty_chart("No performance data available")
            
        # Create heatmap
        fig = px.imshow(
            data,
            title=title,
            height=height,
            color_continuous_scale="RdBu_r",
            aspect="auto",
            template=self.default_template
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Period",
            yaxis_title="Asset",
            coloraxis_colorbar=dict(
                title="Return (%)"
            )
        )
        
        # Update hover template
        fig.update_traces(
            hovertemplate="Asset: %{y}<br>Period: %{x}<br>Return: %{z:.2f}%<extra></extra>"
        )
        
        return fig
    
    def create_correlation_matrix(
        self, 
        data: pd.DataFrame, 
        title: str = None,
        height: int = None
    ) -> go.Figure:
        """
        Create an interactive correlation matrix.
        
        Args:
            data: DataFrame with return data
            title: Chart title
            height: Chart height in pixels
            
        Returns:
            Plotly figure object
        """
        if height is None:
            height = self.default_height
            
        if data.empty:
            return self._create_empty_chart("No correlation data available")
            
        # Calculate correlation matrix
        corr = data.corr()
        
        # Create heatmap
        fig = px.imshow(
            corr,
            title=title,
            height=height,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            template=self.default_template
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Asset",
            yaxis_title="Asset",
            coloraxis_colorbar=dict(
                title="Correlation"
            )
        )
        
        # Update hover template
        fig.update_traces(
            hovertemplate="Asset 1: %{y}<br>Asset 2: %{x}<br>Correlation: %{z:.2f}<extra></extra>"
        )
        
        return fig
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with a message"""
        fig = go.Figure()
        
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        
        fig.update_layout(
            height=self.default_height,
            template=self.default_template,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def render_chart(self, fig: go.Figure, use_container_width: bool = True, key: str = None):
        """
        Render a chart in Streamlit
        
        Args:
            fig: Plotly figure to render
            use_container_width: Whether to use the full container width
            key: Optional unique key for the chart to avoid duplicate ID errors
        """
        st.plotly_chart(fig, use_container_width=use_container_width, key=key)
    
    def render_metrics(
        self, 
        portfolio_value: float, 
        portfolio_change: float,
        benchmark_values: Dict[str, float],
        columns: List[int] = None
    ):
        """
        Render performance metrics in Streamlit.
        
        Args:
            portfolio_value: Current portfolio value
            portfolio_change: Portfolio percentage change
            benchmark_values: Dictionary of benchmark percentage changes
            columns: List of column widths for layout
        """
        if columns is None:
            columns = [2] + [1] * len(benchmark_values)
            
        cols = st.columns(columns)
        
        with cols[0]:
            # Ensure portfolio_change is a valid number
            if pd.isna(portfolio_change) or not np.isfinite(portfolio_change):
                portfolio_change = 0.0
                
            delta_color = "normal" if portfolio_change >= 0 else "inverse"
            st.metric(
                "My Portfolio",
                f"${portfolio_value:,.2f}",
                f"{portfolio_change:.2f}%",
                delta_color=delta_color
            )
        
        for i, (label, value) in enumerate(benchmark_values.items()):
            with cols[i+1]:
                # Ensure value is a valid number
                if pd.isna(value) or not np.isfinite(value):
                    value = 0.0
                    
                delta_color = "normal" if value >= 0 else "inverse"
                st.metric(
                    label,
                    "",
                    f"{value:.2f}%",
                    delta_color=delta_color
                )
    
    def render_alerts(self, insights: List[str], max_visible: int = 5):
        """
        Render alerts with collapsible "Show More" option.
        
        Args:
            insights: List of insight strings
            max_visible: Maximum number of alerts to show initially
        """
        if not insights:
            st.success("🎉 No alerts - portfolio looks healthy!")
            return
            
        # Sort insights by priority
        priority = {"⚠️": 1, "🪙": 2, "🔻": 3, "🚀": 4, "📅": 5}
        insights.sort(key=lambda x: priority.get(x[:2], 6))
        
        # Display top alerts
        for note in insights[:max_visible]:
            st.markdown(f"- {note}")
            
        # Show more if needed
        if len(insights) > max_visible:
            with st.expander(f"Show {len(insights) - max_visible} More Alerts"):
                # If we have a lot of alerts, paginate them
                if len(insights) - max_visible > 15:
                    page_size = 15
                    page_number = st.number_input(
                        "Page", 
                        min_value=1, 
                        max_value=(len(insights) - max_visible + page_size - 1) // page_size,
                        value=1
                    )
                    
                    start_idx = max_visible + (page_number - 1) * page_size
                    end_idx = min(start_idx + page_size, len(insights))
                    
                    for note in insights[start_idx:end_idx]:
                        st.markdown(f"- {note}")
                        
                    st.caption(f"Showing {start_idx+1}-{end_idx} of {len(insights)} alerts")
                else:
                    for note in insights[max_visible:]:
                        st.markdown(f"- {note}")
        
        # Download button
        st.download_button(
            "💾 Download All Alerts", 
            "\n".join(insights), 
            file_name=f"portfolio_alerts_{datetime.date.today()}.txt"
        )

# Create a singleton instance
visualization_helper = VisualizationHelper()