import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import main modules 
# (assuming resource_metrics_and_visualizations.py and paper_visualizations.py are in the same directory)
from resource_metrics_and_visualizations import (
    ResourceMetrics, 
    ModelComparison, 
    add_metrics_to_predictor,
    measure_inference_time, 
    measure_model_parameters
)
from paper_visualizations import PaperVisualizations

def run_tte_prediction_with_metrics(operational_path, tte_path, config_path=None):
    """
    Enhanced version of run_tte_prediction that includes resource metrics and paper visualizations
    
    Parameters:
    - operational_path: Path to operational data CSV
    - tte_path: Path to TTE data CSV
    - config_path: Optional path to config file
    
    Returns:
    - Tuple of (predictor, model_comparison, paper_visualization_figures)
    """
    from tte_predictor import Config, TTEPredictor  # Import your original classes
    
    # 1. Create timestamp and directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = Config(config_path)
    
    # Add additional configuration options
    if not hasattr(config, 'metrics_dir'):
        config.metrics_dir = f"./metrics_{timestamp}"
    if not hasattr(config, 'paper_figures_dir'):
        config.paper_figures_dir = f"./paper_figures_{timestamp}"
    
    # Create directories
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.metrics_dir, exist_ok=True)
    os.makedirs(config.paper_figures_dir, exist_ok=True)
    
    # 2. Initialize the resource metrics tracker
    resource_metrics = ResourceMetrics()
    
    # 3. Create a TTEPredictor instance
    predictor = TTEPredictor(config)
    
    # 4. Enhance the predictor with metrics capabilities
    predictor = add_metrics_to_predictor(predictor)
    
    # 5. Start the overall timer
    resource_metrics.start_monitoring("total", "execution")
    
    # 6. Load and process data with resource monitoring
    resource_metrics.start_monitoring("data_loading", "preprocessing")
    predictor.load_data(operational_path, tte_path)
    resource_metrics.stop_monitoring("data_loading", "preprocessing")
    
    resource_metrics.start_monitoring("cleaning", "preprocessing")
    predictor.clean_data()
    resource_metrics.stop_monitoring("cleaning", "preprocessing")
    
    resource_metrics.start_monitoring("feature_engineering", "preprocessing")
    predictor.engineer_features()
    resource_metrics.stop_monitoring("feature_engineering", "preprocessing")
    
    # 7. Run full analysis (this now includes resource monitoring)
    predictor.run_full_analysis()
    
    # 8. Measure inference metrics
    predictor.measure_inference_metrics(predictor.data)
    
    # 9. Generate comprehensive metrics report
    metrics_report_path = os.path.join(config.metrics_dir, "model_comparison_report")
    predictor.generate_model_comparison_report(metrics_report_path)
    
    # 10. Generate publication-quality figures
    paper_figures = generate_paper_figures(predictor, config.paper_figures_dir)
    
    # 11. Stop overall timer
    resource_metrics.stop_monitoring("total", "execution")
    
    # 12. Print resource usage summary
    resource_metrics.summarize()
    
    print(f"\nAnalysis complete.")
    print(f"Results saved to: {config.results_dir}")
    print(f"Metrics saved to: {config.metrics_dir}")
    print(f"Paper figures saved to: {config.paper_figures_dir}")
    
    return predictor, predictor.model_comparison, paper_figures

def generate_paper_figures(predictor, output_dir):
    """Generate paper-quality figures using the Paper Visualizations module"""
    # Create paper visualizations object
    paper_viz = PaperVisualizations(predictor.model_comparison, figure_path=output_dir)
    
    # Generate all figures
    figures = paper_viz.create_all_figures(predictor.scenario_results)
    
    return figures

def calculate_additional_metrics(predictor):
    """Calculate additional metrics relevant for the paper"""
    results = {}
    
    # 1. Calculate training time ratio (LSTM vs Ridge)
    lstm_time = predictor.resource_metrics.get_metric("LSTM", "training", "duration")
    ridge_time = predictor.resource_metrics.get_metric("Ridge", "training", "duration")
    
    if lstm_time and ridge_time and ridge_time > 0:
        results['lstm_ridge_time_ratio'] = lstm_time / ridge_time
    
    # 2. Calculate memory usage ratio (LSTM vs Ridge)
    lstm_memory = predictor.resource_metrics.get_metric("LSTM", "training", "memory_increase")
    ridge_memory = predictor.resource_metrics.get_metric("Ridge", "training", "memory_increase")
    
    if lstm_memory and ridge_memory and ridge_memory > 0:
        results['lstm_ridge_memory_ratio'] = lstm_memory / ridge_memory
    
    # 3. Calculate accuracy difference (LSTM - Ridge) for each variability
    comparison_df = predictor.model_comparison.create_comparison_table()
    
    for var in comparison_df['Variability'].unique():
        var_df = comparison_df[comparison_df['Variability'] == var]
        
        lstm_mae = var_df.loc[var_df['Model'] == 'LSTM', 'mae'].values
        ridge_mae = var_df.loc[var_df['Model'] == 'Ridge', 'mae'].values
        
        if len(lstm_mae) > 0 and len(ridge_mae) > 0:
            diff = lstm_mae[0] - ridge_mae[0]
            results[f'lstm_ridge_mae_diff_{var}'] = diff
            
            # Calculate percentage improvement
            if ridge_mae[0] > 0:
                pct_improvement = (ridge_mae[0] - lstm_mae[0]) / ridge_mae[0] * 100
                results[f'lstm_ridge_pct_improvement_{var}'] = pct_improvement
    
    # 4. Calculate inference speedup (Ridge vs LSTM)
    lstm_latency = predictor.model_comparison.inference_metrics.get('LSTM', {}).get('latency')
    ridge_latency = predictor.model_comparison.inference_metrics.get('Ridge', {}).get('latency')
    
    if lstm_latency and ridge_latency and lstm_latency > 0:
        results['ridge_lstm_speedup'] = lstm_latency / ridge_latency
    
    # 5. Calculate if Ridge is on the Pareto frontier
    pareto_models = get_pareto_frontier_models(predictor.model_comparison)
    results['ridge_on_pareto'] = 'Ridge' in pareto_models
    
    # Save results to CSV
    results_df = pd.DataFrame([results])
    results_df.to_csv(f"{predictor.config.metrics_dir}/paper_metrics.csv", index=False)
    
    return results

def get_pareto_frontier_models(model_comparison):
    """Identify which models are on the Pareto frontier"""
    comparison_df = model_comparison.create_comparison_table()
    
    if 'mae' not in comparison_df.columns or 'Inference_latency' not in comparison_df.columns:
        return []
    
    pareto_models = []
    
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
            pareto_models.append(row['Model'])
    
    return pareto_models

def generate_paper_summary(predictor, output_file="paper_summary.md"):
    """Generate a summary of key findings for the paper"""
    # Calculate additional metrics
    metrics = calculate_additional_metrics(predictor)
    
    # Get model comparison data
    comparison_df = predictor.model_comparison.create_comparison_table()
    
    # Create the summary markdown
    with open(output_file, 'w') as f:
        f.write("# TTE Prediction Model Comparison: Key Findings\n\n")
        
        f.write("## Performance vs. Computational Cost\n\n")
        
        # Performance metrics by variability
        f.write("### Performance by Variability Level\n\n")
        f.write("| Model | Variability | MAE | RMSE | R² |\n")
        f.write("|-------|------------|-----|------|----|\n")
        
        for _, row in comparison_df.sort_values(['Variability', 'Model']).iterrows():
            mae = row.get('mae', 'N/A')
            if isinstance(mae, (int, float)):
                mae = f"{mae:.4f}"
                
            rmse = row.get('rmse', 'N/A')  
            if isinstance(rmse, (int, float)):
                rmse = f"{rmse:.4f}"
                
            r2 = row.get('r2', 'N/A')
            if isinstance(r2, (int, float)):
                r2 = f"{r2:.4f}"
                
            f.write(f"| {row['Model']} | {row['Variability']} | {mae} | {rmse} | {r2} |\n")
        
        f.write("\n### Computational Efficiency\n\n")
        f.write("| Model | Inference Time (s) | Memory Usage (MB) | Parameters |\n")
        f.write("|-------|---------------------|-------------------|------------|\n")
        
        for model in comparison_df['Model'].unique():
            model_rows = comparison_df[comparison_df['Model'] == model]
            if not model_rows.empty:
                row = model_rows.iloc[0]
                
                inference_time = row.get('Inference_latency', 'N/A')
                if isinstance(inference_time, (int, float)):
                    inference_time = f"{inference_time:.5f}"
                    
                memory = row.get('Memory_increase', 'N/A')
                if isinstance(memory, (int, float)):
                    memory = f"{memory:.1f}"
                    
                params = row.get('Parameters', 'N/A')
                if isinstance(params, (int, float)):
                    params = f"{params:.0f}"
                    
                f.write(f"| {model} | {inference_time} | {memory} | {params} |\n")
        
        # Key comparative metrics
        f.write("\n## Key Comparative Metrics\n\n")
        
        if 'lstm_ridge_time_ratio' in metrics:
            f.write(f"- **Training Time Ratio (LSTM:Ridge)**: {metrics['lstm_ridge_time_ratio']:.2f}x\n")
        
        if 'lstm_ridge_memory_ratio' in metrics:
            f.write(f"- **Memory Usage Ratio (LSTM:Ridge)**: {metrics['lstm_ridge_memory_ratio']:.2f}x\n")
        
        if 'ridge_lstm_speedup' in metrics:
            f.write(f"- **Inference Speedup (Ridge vs LSTM)**: {metrics['ridge_lstm_speedup']:.2f}x\n")
        
        # Performance differences by variability
        f.write("\n### Performance Differences by Variability\n\n")
        
        for var in ['low', 'medium', 'high']:
            diff_key = f'lstm_ridge_mae_diff_{var}'
            pct_key = f'lstm_ridge_pct_improvement_{var}'
            
            if diff_key in metrics:
                diff = metrics[diff_key]
                if diff < 0:
                    better = "LSTM outperforms Ridge"
                    diff_abs = abs(diff)
                else:
                    better = "Ridge outperforms LSTM"
                    diff_abs = diff
                
                f.write(f"- **{var.capitalize()} Variability**: {better} by MAE difference of {diff_abs:.4f}")
                
                if pct_key in metrics:
                    pct = metrics[pct_key]
                    if pct > 0:
                        f.write(f" ({pct:.1f}% improvement)\n")
                    else:
                        f.write(f" ({abs(pct):.1f}% worse)\n")
                else:
                    f.write("\n")
        
        # Pareto efficiency
        f.write("\n### Pareto Efficiency\n\n")
        
        pareto_models = get_pareto_frontier_models(predictor.model_comparison)
        if pareto_models:
            f.write("Models on the Pareto frontier (optimal trade-off between performance and computational cost):\n\n")
            for model in pareto_models:
                f.write(f"- {model}\n")
        else:
            f.write("No models identified on the Pareto frontier.\n")
        
        # Main findings
        f.write("\n## Main Findings\n\n")
        
        # Automatically determine if Ridge is competitive
        ridge_competitive = False
        lstm_better_pct = 0
        
        for var in ['low', 'medium', 'high']:
            pct_key = f'lstm_ridge_pct_improvement_{var}'
            if pct_key in metrics:
                # If LSTM is less than 10% better than Ridge in any scenario, 
                # we consider Ridge competitive
                if metrics[pct_key] < 10:
                    ridge_competitive = True
                else:
                    lstm_better_pct = max(lstm_better_pct, metrics[pct_key])
        
        if ridge_competitive:
            f.write("1. **Ridge Regression is a competitive low-fidelity alternative** to LSTM for TTE prediction, ")
            f.write("offering comparable performance with significantly lower computational requirements.\n\n")
        else:
            f.write(f"1. **LSTM outperforms Ridge Regression significantly** (up to {lstm_better_pct:.1f}% better) ")
            f.write("but requires substantially more computational resources.\n\n")
        
        if 'ridge_lstm_speedup' in metrics:
            f.write(f"2. **Ridge Regression is {metrics['ridge_lstm_speedup']:.1f}x faster for inference** than LSTM, ")
            f.write("making it suitable for real-time or resource-constrained applications.\n\n")
        
        if 'ridge_on_pareto' in metrics and metrics['ridge_on_pareto']:
            f.write("3. **Ridge Regression lies on the Pareto frontier**, indicating it offers an optimal trade-off ")
            f.write("between prediction performance and computational efficiency.\n\n")
        
        # Variability-specific findings
        best_var_for_ridge = None
        worst_var_for_ridge = None
        best_diff = float('inf')
        worst_diff = float('-inf')
        
        for var in ['low', 'medium', 'high']:
            diff_key = f'lstm_ridge_mae_diff_{var}'
            if diff_key in metrics:
                if metrics[diff_key] < best_diff:
                    best_diff = metrics[diff_key]
                    best_var_for_ridge = var
                
                if metrics[diff_key] > worst_diff:
                    worst_diff = metrics[diff_key]
                    worst_var_for_ridge = var
        
        if best_var_for_ridge:
            f.write(f"4. **Ridge performs best on {best_var_for_ridge} variability data**, ")
            
            if best_diff < 0:
                f.write(f"where it actually outperforms LSTM by {abs(best_diff):.4f} MAE.\n\n")
            else:
                f.write(f"where the performance gap with LSTM is smallest ({best_diff:.4f} MAE).\n\n")
        
        if worst_var_for_ridge and worst_var_for_ridge != best_var_for_ridge:
            f.write(f"5. **Ridge struggles most with {worst_var_for_ridge} variability data**, ")
            f.write(f"where LSTM outperforms it by {abs(worst_diff):.4f} MAE.\n\n")
        
        # Resource efficiency summary
        if 'lstm_ridge_time_ratio' in metrics and 'lstm_ridge_memory_ratio' in metrics:
            f.write(f"6. **Resource efficiency**: Ridge Regression requires {1/metrics['lstm_ridge_time_ratio']:.1f}x less training time ")
            f.write(f"and {1/metrics['lstm_ridge_memory_ratio']:.1f}x less memory than LSTM, ")
            f.write("making it suitable for deployment in environments with limited computational resources.\n\n")
        
        f.write("## Conclusion\n\n")
        
        if ridge_competitive:
            f.write("This study demonstrates that simple, low-fidelity models like Ridge Regression ")
            f.write("can effectively replace more complex deep learning approaches for TTE prediction ")
            f.write("in many practical scenarios, especially when computational resources are limited ")
            f.write("or when rapid training and inference are required. The significant reduction in ")
            f.write("computational complexity comes with only a modest decrease in prediction accuracy, ")
            f.write("making Ridge Regression a compelling alternative for many real-world applications.")
        else:
            f.write("This study shows that while deep learning approaches like LSTM offer superior ")
            f.write("prediction accuracy for TTE prediction, simpler models like Ridge Regression ")
            f.write("still provide reasonable performance with dramatically lower computational requirements. ")
            f.write("The choice between these approaches depends on the specific application requirements, ")
            f.write("with Ridge Regression being particularly suitable for resource-constrained environments ")
            f.write("where computational efficiency is prioritized over maximum prediction accuracy.")
    
    print(f"Paper summary generated: {output_file}")
    return output_file

# -------------- Example Usage --------------

if __name__ == "__main__":
    # Example configuration
    config = {
        'vehicle_subset_size': 250,     # Use fewer vehicles to reduce computation
        'skip_lstm': False,             # Include LSTM for comparison
        'feature_selection': {
            'enabled': True,
            'k': 15                     # Limit to top 15 features
        },
        'sequence_limit': 1000,         # Limit sequences per fold
    }
    
    # Save config to file
    import json
    with open('tte_paper_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Run the enhanced TTE prediction pipeline
    predictor, model_comparison, figures = run_tte_prediction_with_metrics(
        './2241data/train_operational_readouts.csv',
        './2241data/train_tte.csv',
        config_path='tte_paper_config.json'
    )
    
    # Generate paper summary
    summary_file = generate_paper_summary(predictor, "tte_prediction_paper_findings.md")
    
    print(f"\nAnalysis complete. Paper materials generated.")
    print(f"Review the summary at: {summary_file}")