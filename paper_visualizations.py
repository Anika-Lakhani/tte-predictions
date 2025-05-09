import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import matplotlib.lines as mlines

class PaperVisualizations:
    """
    Create publication-quality visualizations for an academic paper on
    low-fidelity vs. high-fidelity TTE prediction methods.
    """
    def __init__(self, model_comparison, figure_path="figures"):
        """
        Initialize with a ModelComparison instance to access metrics data
        
        Parameters:
        - model_comparison: ModelComparison instance with model metrics
        - figure_path: Path to save the figures
        """
        self.model_comparison = model_comparison
        self.figure_path = figure_path
        
        # Set unified style for all plots
        self.set_style()
        
        # Create color scheme for models
        self.model_colors = {
            'LSTM': '#1f77b4',      # Blue
            'Ridge': '#ff7f0e',     # Orange
            'ES': '#2ca02c',        # Green
            'Ensemble': '#d62728'   # Red
        }
        
        # Create markers for variability levels
        self.variability_markers = {
            'low': 'o',
            'medium': 's',
            'high': '^'
        }
    
    def set_style(self):
        """Set unified style for publication quality plots"""
        sns.set_style("whitegrid")
        # plt.rcParams['font.family'] = 'serif'
        # plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 16
    
    def fig1_performance_vs_complexity(self):
        """
        Create Figure 1: Performance (MAE) vs. Model Complexity
        
        This is the central figure showing the relationship between model 
        complexity and prediction performance across variability scenarios.
        """
        # Prepare data
        comparison_df = self.model_comparison.create_comparison_table()
        
        if 'mae' not in comparison_df.columns:
            print("Missing required metrics for performance vs. complexity plot")
            return None
        
        # Add parameter counts if not already in the data
        if 'Parameters' not in comparison_df.columns:
            model_params = {
                'LSTM': 5000,     # Approximate for a small LSTM
                'Ridge': 50,      # Approximate for Ridge model
                'ES': 1,          # ES has just the smoothing parameter
                'Ensemble': 52    # Combined parameters from Ridge and ES
            }
            comparison_df['Parameters'] = comparison_df['Model'].map(model_params)
        
        # Create figure with 2 subplots: main plot and bar chart
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        
        # Main scatter plot (MAE vs. Parameters)
        ax1 = plt.subplot(gs[0])
        
        # Plot each point
        for model in comparison_df['Model'].unique():
            for var in comparison_df['Variability'].unique():
                df_subset = comparison_df[(comparison_df['Model'] == model) & 
                                        (comparison_df['Variability'] == var)]
                if not df_subset.empty:
                    ax1.scatter(
                        df_subset['Parameters'], 
                        df_subset['mae'],
                        s=150,
                        marker=self.variability_markers.get(var.lower(), 'o'),
                        color=self.model_colors.get(model, 'gray'),
                        label=f"{model} ({var})"
                    )
        
        # Add fit line to show general trend
        try:
            x = np.log10(comparison_df['Parameters'])
            y = comparison_df['mae']
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(x), max(x), 100)
            ax1.plot(10**x_range, p(x_range), '--', color='black', alpha=0.5)
        except:
            pass  # Skip trendline if fitting fails
        
        # Format axes
        ax1.set_xscale('log')
        ax1.set_xlabel('Number of Model Parameters (log scale)')
        ax1.set_ylabel('Mean Absolute Error (MAE)')
        ax1.set_title('Performance vs. Model Complexity')
        
        # Create custom legend
        model_handles = [Patch(color=self.model_colors[m], label=m) 
                       for m in self.model_colors if m in comparison_df['Model'].unique()]
        var_handles = [mlines.Line2D([], [], color='black', marker=self.variability_markers[v], 
                                  linestyle='None', markersize=8, label=v.capitalize())
                    for v in self.variability_markers if v in [var.lower() for var in comparison_df['Variability'].unique()]]
        
        # Combine legends
        ax1.legend(handles=model_handles + var_handles, 
                loc='upper center', bbox_to_anchor=(0.5, -0.15),
                ncol=len(model_handles) + len(var_handles), frameon=False)
        
        # Add annotations to points
        for _, row in comparison_df.iterrows():
            ax1.annotate(
                f"{row['Model']}\n({row['Variability']})",
                (row['Parameters'], row['mae']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )
        
        # Bar chart showing relative parameter counts
        ax2 = plt.subplot(gs[1])
        
        # Aggregate parameter counts by model (maximum across scenarios)
        model_params = comparison_df.groupby('Model')['Parameters'].max().reset_index()
        
        # Create bar chart
        bar_positions = np.arange(len(model_params))
        bars = ax2.bar(
            bar_positions,
            model_params['Parameters'],
            color=[self.model_colors.get(m, 'gray') for m in model_params['Model']]
        )
        
        # Format bar chart
        ax2.set_yscale('log')
        ax2.set_xticks(bar_positions)
        ax2.set_xticklabels(model_params['Model'])
        ax2.set_ylabel('Parameters\n(log scale)')
        
        # Add parameter counts on top of bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width()/2.,
                height * 1.1,
                f"{height:.0f}",
                ha='center',
                fontsize=9
            )
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4)
        
        # Save figure
        plt.savefig(f"{self.figure_path}/fig1_performance_vs_complexity.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.figure_path}/fig1_performance_vs_complexity.pdf", bbox_inches='tight')
        
        return fig
    
    def fig2_computational_efficiency(self):
        """
        Create Figure 2: Computational Efficiency Metrics
        
        Shows inference time, memory usage, and throughput for each model.
        """
        # Get data
        comparison_df = self.model_comparison.create_comparison_table()
        
        # Check if required metrics are available
        required_metrics = ['Inference_latency', 'Memory_increase']
        missing_metrics = [m for m in required_metrics if m not in comparison_df.columns]
        if missing_metrics:
            print(f"Missing required metrics: {missing_metrics}")
            return None
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Aggregate by model (take mean across variability levels)
        model_metrics = comparison_df.groupby('Model').agg({
            'Inference_latency': 'mean',
            'Memory_increase': 'mean',
            'Parameters': 'mean'  # Use mean parameters
        }).reset_index()
        
        # Calculate throughput (samples/sec) from latency if not provided
        if 'Inference_throughput' not in model_metrics.columns:
            # Assume a standard batch size of 1000 samples
            batch_size = 1000
            model_metrics['Throughput'] = batch_size / model_metrics['Inference_latency']
        
        # Ensure at least some data is available
        if model_metrics.empty:
            print("No model metrics available for computational efficiency plot")
            return None
        
        # 1. Inference Time (Latency)
        ax1 = axes[0]
        bars1 = ax1.bar(
            model_metrics['Model'],
            model_metrics['Inference_latency'],
            color=[self.model_colors.get(m, 'gray') for m in model_metrics['Model']]
        )
        ax1.set_ylabel('Inference Time (seconds)')
        ax1.set_title('Model Inference Latency')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width()/2.,
                height * 1.05,
                f"{height:.4f}",
                ha='center',
                fontsize=9
            )
        
        # 2. Memory Usage
        ax2 = axes[1]
        bars2 = ax2.bar(
            model_metrics['Model'],
            model_metrics['Memory_increase'],
            color=[self.model_colors.get(m, 'gray') for m in model_metrics['Model']]
        )
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Model Memory Footprint')
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width()/2.,
                height * 1.05,
                f"{height:.1f}",
                ha='center',
                fontsize=9
            )
        
        # 3. Throughput
        ax3 = axes[2]
        bars3 = ax3.bar(
            model_metrics['Model'],
            model_metrics['Throughput'],
            color=[self.model_colors.get(m, 'gray') for m in model_metrics['Model']]
        )
        ax3.set_ylabel('Samples per Second')
        ax3.set_title('Model Throughput')
        
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width()/2.,
                height * 1.05,
                f"{height:.1f}",
                ha='center',
                fontsize=9
            )
        
        # Add a suptitle
        plt.suptitle('Computational Efficiency Metrics', fontsize=16)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{self.figure_path}/fig2_computational_efficiency.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.figure_path}/fig2_computational_efficiency.pdf", bbox_inches='tight')
        
        return fig
    
    def fig3_performance_by_variability(self):
        """
        Create Figure 3: Performance across Variability Levels
        
        Shows how each model performs across different variability scenarios.
        """
        # Get data
        comparison_df = self.model_comparison.create_comparison_table()
        
        if 'mae' not in comparison_df.columns:
            print("Missing required metrics for performance by variability plot")
            return None
        
        # Ensure we have multiple variability levels
        if comparison_df['Variability'].nunique() < 2:
            print("Insufficient variability levels for comparison")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Define width of bars
        n_models = comparison_df['Model'].nunique()
        width = 0.8 / n_models
        
        # Plot grouped bars for each model
        models = comparison_df['Model'].unique()
        variabilities = sorted(comparison_df['Variability'].unique())
        
        x = np.arange(len(variabilities))
        
        for i, model in enumerate(models):
            model_data = []
            for var in variabilities:
                subset = comparison_df[(comparison_df['Model'] == model) & 
                                     (comparison_df['Variability'] == var)]
                if not subset.empty:
                    model_data.append(subset['mae'].values[0])
                else:
                    model_data.append(np.nan)
            
            # Calculate position for this model's bars
            pos = x + width * (i - n_models / 2 + 0.5)
            
            # Plot bars
            bars = ax.bar(
                pos, 
                model_data,
                width,
                label=model,
                color=self.model_colors.get(model, 'gray')
            )
            
            # Add value labels
            for bar, val in zip(bars, model_data):
                if not np.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        bar.get_height() * 1.05,
                        f"{val:.4f}",
                        ha='center',
                        fontsize=8
                    )
        
        # Format axes
        ax.set_xticks(x)
        ax.set_xticklabels([v.capitalize() for v in variabilities])
        ax.set_ylabel('Mean Absolute Error (MAE)')
        ax.set_xlabel('Data Variability Level')
        ax.set_title('Model Performance Across Variability Scenarios')
        ax.legend()
        
        # Add grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{self.figure_path}/fig3_performance_by_variability.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.figure_path}/fig3_performance_by_variability.pdf", bbox_inches='tight')
        
        return fig
    
    def fig4_efficiency_frontier(self):
        """
        Create Figure 4: Efficiency Frontier
        
        Shows the trade-off between performance and computational cost
        with Pareto frontier highlighting optimal models.
        """
        # Get data
        comparison_df = self.model_comparison.create_comparison_table()
        
        # Check if required metrics are available
        if 'mae' not in comparison_df.columns or 'Inference_latency' not in comparison_df.columns:
            print("Missing required metrics for efficiency frontier plot")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot each point with custom markers for variability
        for model in comparison_df['Model'].unique():
            for var in comparison_df['Variability'].unique():
                df_subset = comparison_df[(comparison_df['Model'] == model) & 
                                        (comparison_df['Variability'] == var)]
                if not df_subset.empty:
                    ax.scatter(
                        df_subset['Inference_latency'], 
                        df_subset['mae'],
                        s=120,
                        marker=self.variability_markers.get(var.lower(), 'o'),
                        color=self.model_colors.get(model, 'gray'),
                        label=f"{model} ({var})"
                    )
        
        # Add labels to each point
        for _, row in comparison_df.iterrows():
            ax.annotate(
                f"{row['Model']}\n({row['Variability']})",
                (row['Inference_latency'], row['mae']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )
        
        # Calculate and plot Pareto frontier
        pareto_points = []
        for i, row in comparison_df.iterrows():
            is_pareto = True
            for j, other_row in comparison_df.iterrows():
                if i != j:
                    if (other_row['Inference_latency'] <= row['Inference_latency'] and 
                        other_row['mae'] <= row['mae'] and
                        (other_row['Inference_latency'] < row['Inference_latency'] or 
                         other_row['mae'] < row['mae'])):
                        is_pareto = False
                        break
            if is_pareto:
                pareto_points.append((row['Inference_latency'], row['mae']))
        
        # Sort points for line plotting
        pareto_points.sort()
        if pareto_points:
            pareto_x, pareto_y = zip(*pareto_points)
            ax.plot(pareto_x, pareto_y, 'r--', linewidth=2, label='Pareto Frontier')
        
        # Format axes
        ax.set_xscale('log')  # Log scale for latency
        ax.set_xlabel('Inference Time (seconds, log scale)')
        ax.set_ylabel('Mean Absolute Error (MAE)')
        ax.set_title('Efficiency Frontier: Performance vs. Computational Cost')
        
        # Add the "Ideal Point" in the bottom-left corner
        ideal_x = min(comparison_df['Inference_latency']) * 0.9
        ideal_y = min(comparison_df['mae']) * 0.9
        ax.scatter(ideal_x, ideal_y, marker='*', color='red', s=200, label='Ideal Point')
        
        # Add shaded regions for "efficient" and "inefficient" areas
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        
        # Add custom legend with both model colors and variability markers
        handles, labels = ax.get_legend_handles_labels()
        
        # Remove duplicates (keeping order)
        by_label = dict(zip(labels, handles))
        
        # Get unique model and variability markers for legend
        model_handles = [Patch(color=self.model_colors[m], label=m) 
                       for m in self.model_colors if m in comparison_df['Model'].unique()]
        var_handles = [mlines.Line2D([], [], color='black', marker=self.variability_markers[v], 
                                  linestyle='None', markersize=8, label=v.capitalize())
                    for v in self.variability_markers if v in [var.lower() for var in comparison_df['Variability'].unique()]]
        
        # Add Pareto frontier and ideal point to legend
        special_handles = []
        if pareto_points:
            special_handles.append(mlines.Line2D([], [], color='red', linestyle='--', 
                                            label='Pareto Frontier'))
        special_handles.append(mlines.Line2D([], [], marker='*', color='red', 
                                         linestyle='None', markersize=10, 
                                         label='Ideal Point'))
        
        # Create combined legend
        ax.legend(handles=model_handles + var_handles + special_handles, 
                loc='upper right', frameon=True, framealpha=0.9)
        
        # Add explanatory annotations
        ax.annotate(
            "BETTER PERFORMANCE",
            xy=(x_max*0.5, y_min*1.1),
            xytext=(x_max*0.5, y_min*1.1),
            ha='center',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
        
        ax.annotate(
            "BETTER EFFICIENCY",
            xy=(x_min*1.1, y_max*0.5),
            xytext=(x_min*1.1, y_max*0.5),
            ha='left',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
        
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{self.figure_path}/fig4_efficiency_frontier.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.figure_path}/fig4_efficiency_frontier.pdf", bbox_inches='tight')
        
        return fig
    
    def fig5_prediction_comparison(self, scenario_results, variability='medium'):
        """
        Create Figure 5: Prediction Comparison
        
        Shows actual vs. predicted values for different models on a sample
        of test data, with confidence intervals.
        
        Parameters:
        - scenario_results: Results dictionary from TTEPredictor for a specific variability
        - variability: Variability level to use for the plot
        """
        # Check if required data is available
        if 'y_test' not in scenario_results or scenario_results['y_test'] is None:
            print("Missing test data for prediction comparison plot")
            return None
        
        # Get test data and predictions
        y_test = scenario_results['y_test']
        
        # Limit to a smaller subset for clarity
        n_samples = min(50, len(y_test))
        x_indices = np.arange(n_samples)
        y_test = y_test[:n_samples]
        
        # Get predictions for each model
        predictions = {}
        for model, prefix in [
            ('LSTM', 'lstm'),
            ('Ridge', 'lr'),
            ('ES', 'es'),
            ('Ensemble', 'ensemble')
        ]:
            pred_key = f'{prefix}_preds'
            if pred_key in scenario_results and scenario_results[pred_key] is not None:
                predictions[model] = scenario_results[pred_key][:n_samples]
        
        if not predictions:
            print("No prediction data available")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot actual values
        ax.plot(x_indices, y_test, 'k-', linewidth=2.5, label='Actual')
        
        # Plot each model's predictions
        for model, preds in predictions.items():
            ax.plot(
                x_indices, 
                preds, 
                '--', 
                linewidth=1.8, 
                color=self.model_colors.get(model, 'gray'),
                label=f'{model} Predictions'
            )
            
            # Add confidence interval if clean predictions are available
            clean_key = f'{model.lower()}_preds_clean'
            if clean_key in scenario_results and scenario_results[clean_key] is not None:
                # Calculate standard error
                residuals = y_test - scenario_results[clean_key][:n_samples]
                std_err = np.std(residuals)
                
                # Plot confidence intervals (95%)
                ax.fill_between(
                    x_indices,
                    preds - 1.96 * std_err,
                    preds + 1.96 * std_err,
                    alpha=0.2,
                    color=self.model_colors.get(model, 'gray')
                )
        
        # Format axes
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Normalized TTE')
        ax.set_title(f'Model Predictions ({variability.capitalize()} Variability Scenario)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{self.figure_path}/fig5_prediction_comparison_{variability}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.figure_path}/fig5_prediction_comparison_{variability}.pdf", bbox_inches='tight')
        
        return fig
    
    def fig6_feature_importance(self, scenario_results, variability='medium'):
        """
        Create Figure 6: Feature Importance
        
        Shows the most important features for the Ridge model.
        
        Parameters:
        - scenario_results: Results dictionary from TTEPredictor for a specific variability
        - variability: Variability level to use for the plot
        """
        # Check if feature importance data is available
        if 'feature_importance' not in scenario_results:
            print("No feature importance data available")
            return None
        
        # Get feature importance data
        fi_data = scenario_results['feature_importance']
        
        # Limit to top 15 features
        top_n = min(15, len(fi_data))
        top_features = fi_data.head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot horizontal bars
        bars = ax.barh(
            top_features['feature'],
            top_features['importance'],
            color=self.model_colors.get('Ridge', '#ff7f0e')
        )
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width * 1.01,
                bar.get_y() + bar.get_height()/2,
                f"{width:.4f}",
                va='center',
                fontsize=9
            )
        
        # Format axes
        ax.set_xlabel('Feature Importance (Absolute Coefficient Value)')
        ax.set_title(f'Top Feature Importance: Ridge Model ({variability.capitalize()} Variability)')
        ax.invert_yaxis()  # Most important at the top
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{self.figure_path}/fig6_feature_importance_{variability}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.figure_path}/fig6_feature_importance_{variability}.pdf", bbox_inches='tight')
        
        return fig
    
    def fig7_radar_chart(self):
        """
        Create Figure 7: Radar Chart
        
        Shows a comprehensive comparison of models across multiple metrics
        using a radar/spider chart.
        """
        # Get data
        comparison_df = self.model_comparison.create_comparison_table()
        
        # We'll focus on the medium variability scenario for clarity
        if 'Variability' in comparison_df.columns:
            # Check which variability levels are available
            var_levels = comparison_df['Variability'].unique()
            
            if 'medium' in var_levels:
                selected_var = 'medium'
            elif len(var_levels) > 0:
                # Use the first available level
                selected_var = var_levels[0]
            else:
                print("No variability data available")
                return None
            
            df = comparison_df[comparison_df['Variability'] == selected_var]
        else:
            df = comparison_df
        
        if df.empty:
            print("No data available for radar chart")
            return None
        
        # Define the metrics to include in the radar chart
        metrics = [
            ('mae', 'Accuracy (MAE)', True),  # Name, Label, is_lower_better
            ('r2', 'Explanatory Power (R²)', False),
            ('Inference_latency', 'Speed', True),
            ('Memory_increase', 'Memory Efficiency', True),
            ('Parameters', 'Model Simplicity', True)
        ]
        
        # Keep only available metrics
        metrics = [(name, label, is_lower_better) for name, label, is_lower_better in metrics 
                 if name in df.columns]
        
        if len(metrics) < 3:
            print("Insufficient metrics for radar chart (need at least 3)")
            return None
        
        # Normalize data for radar chart (0-1, higher is better)
        models = df['Model'].unique()
        radar_data = []
        
        for model in models:
            model_data = []
            model_row = df[df['Model'] == model].iloc[0]
            
            for metric_name, _, is_lower_better in metrics:
                if metric_name in model_row:
                    # Get all values for this metric
                    all_values = df[metric_name].values
                    min_val = min(all_values)
                    max_val = max(all_values)
                    
                    # Skip if all values are the same
                    if max_val == min_val:
                        model_data.append(1.0)  # Default to perfect score
                        continue
                    
                    # Get this model's value
                    value = model_row[metric_name]
                    
                    # Normalize to 0-1 (higher is better)
                    if is_lower_better:
                        normalized = 1 - ((value - min_val) / (max_val - min_val))
                    else:
                        normalized = (value - min_val) / (max_val - min_val)
                    
                    model_data.append(normalized)
                else:
                    model_data.append(0)  # Missing metric
            
            radar_data.append(model_data)
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Number of metrics
        N = len(metrics)
        
        # Angles for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Labels for axes
        ax.set_theta_offset(np.pi / 2)  # Start at top
        ax.set_theta_direction(-1)  # Go clockwise
        
        # Draw axis lines
        ax.set_thetagrids(np.degrees(angles[:-1]), [label for _, label, _ in metrics])
        
        # Draw y-axis labels
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], fontsize=9)
        plt.ylim(0, 1)
        
        # Plot data
        for i, model in enumerate(models):
            values = radar_data[i]
            values += values[:1]  # Close the loop
            
            ax.plot(
                angles, 
                values, 
                linewidth=2, 
                linestyle='solid', 
                label=model,
                color=self.model_colors.get(model, 'gray')
            )
            ax.fill(
                angles, 
                values, 
                color=self.model_colors.get(model, 'gray'),
                alpha=0.1
            )
        
        # Add title and legend
        var_text = f" ({selected_var.capitalize()} Variability)" if 'Variability' in comparison_df.columns else ""
        plt.title(f'Multi-Metric Model Comparison{var_text}', size=15, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Save figure
        plt.savefig(f"{self.figure_path}/fig7_radar_chart.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.figure_path}/fig7_radar_chart.pdf", bbox_inches='tight')
        
        return fig
    
    def create_all_figures(self, scenario_results=None):
        """
        Create all figures for the paper
        
        Parameters:
        - scenario_results: Dictionary of results from TTEPredictor for specific variabilities
        
        Returns:
        - Dictionary of figure objects
        """
        import os
        os.makedirs(self.figure_path, exist_ok=True)
        
        figures = {}
        
        # Main figures that don't need scenario_results
        figures['fig1'] = self.fig1_performance_vs_complexity()
        figures['fig2'] = self.fig2_computational_efficiency()
        figures['fig3'] = self.fig3_performance_by_variability()
        figures['fig4'] = self.fig4_efficiency_frontier()
        figures['fig7'] = self.fig7_radar_chart()
        
        # Figures that need scenario_results
        if scenario_results:
            # Use medium variability if available
            if 'medium' in scenario_results:
                var = 'medium'
            elif scenario_results:
                # Use the first available variability
                var = list(scenario_results.keys())[0]
            else:
                var = None
            
            if var:
                figures['fig5'] = self.fig5_prediction_comparison(scenario_results[var], var)
                figures['fig6'] = self.fig6_feature_importance(scenario_results[var], var)
        
        return figures


# Function to use with TTEPredictor framework
def generate_paper_figures(tte_predictor, output_dir="paper_figures"):
    """
    Generate publication-quality figures for an academic paper based on TTEPredictor results
    
    Parameters:
    - tte_predictor: Enhanced TTEPredictor instance with model_comparison
    - output_dir: Directory to save the figures
    
    Returns:
    - Dictionary of figure objects
    """
    # Check if predictor has the model_comparison attribute
    if not hasattr(tte_predictor, 'model_comparison'):
        print("TTEPredictor doesn't have model_comparison attribute")
        # Try to create it
        try:
            from resource_metrics_and_visualizations import enhance_tte_predictor_with_metrics
            resource_metrics, model_comparison = enhance_tte_predictor_with_metrics(tte_predictor)
            tte_predictor.model_comparison = model_comparison
        except Exception as e:
            print(f"Error creating model_comparison: {e}")
            return None
    
    # Create paper visualizations
    paper_viz = PaperVisualizations(tte_predictor.model_comparison, figure_path=output_dir)
    
    # Generate figures
    figures = paper_viz.create_all_figures(tte_predictor.scenario_results)
    
    print(f"Generated {len(figures)} publication-quality figures in '{output_dir}'")
    
    return figures