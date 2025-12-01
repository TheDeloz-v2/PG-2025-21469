import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


def calculate_financial_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    prices: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate financial and trading metrics
    
    Args:
        predictions: Predicted returns or prices
        actuals: Actual returns or prices
        prices: Actual prices (if predicting returns)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Filter NaN and inf values
    valid_mask = np.isfinite(predictions) & np.isfinite(actuals)
    if not valid_mask.any():
        # Return NaN metrics if no valid data
        return {
            'rmse': np.nan, 'mae': np.nan, 'mape': np.nan,
            'directional_accuracy': np.nan, 'r2': np.nan,
            'sharpe_ratio': np.nan, 'max_drawdown': np.nan,
            'hit_rate': np.nan, 'profit_factor': np.nan
        }
    
    predictions = predictions[valid_mask]
    actuals = actuals[valid_mask]
    
    # Statistical metrics
    metrics['rmse'] = np.sqrt(mean_squared_error(actuals, predictions))
    metrics['mae'] = mean_absolute_error(actuals, predictions)
    
    # MAPE
    mask = actuals != 0
    if mask.sum() > 0:
        metrics['mape'] = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100
    else:
        metrics['mape'] = np.nan
    
    # Directional accuracy
    # For returns: positive = price went up, negative = price went down
    if len(actuals) > 0:
        actual_direction = np.sign(actuals)
        pred_direction = np.sign(predictions)
        metrics['directional_accuracy'] = np.mean(actual_direction == pred_direction) * 100
    else:
        metrics['directional_accuracy'] = 0.0
    
    # R2
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    if len(predictions) > 0:
        # Strategy: long if predict positive return (up), short if predict negative return (down)
        strategy_returns = np.where(predictions > 0, actuals, -actuals)
        
        # Sharpe Ratio
        if len(strategy_returns) > 0 and strategy_returns.std() != 0:
            metrics['sharpe_ratio'] = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        else:
            metrics['sharpe_ratio'] = 0.0
        
        # Max Drawdown
        cumulative_returns = (1 + strategy_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        metrics['max_drawdown'] = np.min(drawdown) * 100
        
        # Hit Rate
        metrics['hit_rate'] = np.mean(strategy_returns > 0) * 100
        
        # Profit Factor
        profits = strategy_returns[strategy_returns > 0].sum() if (strategy_returns > 0).any() else 0
        losses = -strategy_returns[strategy_returns < 0].sum() if (strategy_returns < 0).any() else 0
        metrics['profit_factor'] = profits / losses if losses != 0 else (np.inf if profits > 0 else 0)
    
    return metrics


def compare_models(
    model_results: Dict[str, Dict],
    metric: str = 'directional_accuracy'
) -> pd.DataFrame:
    """
    Compare multiple models based on metrics
    
    Args:
        model_results: Dictionary of model names to their metrics
        metric: Primary metric for ranking
        
    Returns:
        Comparison DataFrame sorted by metric
    """
    comparison = []
    
    for model_name, metrics in model_results.items():
        row = {'model': model_name}
        row.update(metrics)
        comparison.append(row)
    
    df = pd.DataFrame(comparison)
    ascending = metric in ['rmse', 'mae', 'mape', 'max_drawdown']
    df = df.sort_values(by=metric, ascending=ascending)
    
    return df


def plot_predictions_vs_actual(
    dates: np.ndarray,
    actuals: np.ndarray,
    predictions: Dict[str, np.ndarray],
    title: str = "Predictions vs Actual",
    save_path: Optional[str] = None
):
    """
    Plot predictions from multiple models vs actual values
    
    Args:
        dates: Date array
        actuals: Actual values
        predictions: Dictionary of model_name -> predictions
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(15, 8))
    
    # Plot actual
    plt.plot(dates, actuals, label='Actual', color='black', linewidth=2, alpha=0.8)
    
    # Plot predictions
    colors = plt.cm.Set2(np.linspace(0, 1, len(predictions)))
    for (model_name, preds), color in zip(predictions.items(), colors):
        plt.plot(dates, preds, label=model_name, alpha=0.7, linewidth=1.5, color=color)
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metrics: list = None,
    save_path: Optional[str] = None
):
    """
    Plot bar charts comparing models across metrics
    
    Args:
        comparison_df: DataFrame from compare_models()
        metrics: List of metrics to plot
        save_path: Path to save figure
    """
    if metrics is None:
        metrics = ['rmse', 'mae', 'directional_accuracy', 'sharpe_ratio']
    
    # Filter available metrics
    metrics = [m for m in metrics if m in comparison_df.columns]
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        # Sort
        ascending = metric in ['rmse', 'mae', 'mape', 'max_drawdown']
        sorted_df = comparison_df.sort_values(by=metric, ascending=ascending)
        
        # Plot
        bars = ax.barh(sorted_df['model'], sorted_df[metric])
        
        # Color
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bars))) if ascending else \
                 plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel(metric.upper(), fontsize=11, fontweight='bold')
        ax.set_title(f'{metric.upper()} Comparison', fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        
        # Labels
        for i, (model, value) in enumerate(zip(sorted_df['model'], sorted_df[metric])):
            ax.text(value, i, f' {value:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()


def create_evaluation_report(
    model_results: Dict[str, Dict],
    save_path: Optional[str] = None
) -> str:
    """
    Create comprehensive evaluation report
    
    Args:
        model_results: Dictionary of model_name -> metrics
        save_path: Path to save report
        
    Returns:
        Report string
    """
    report = []
    report.append("="*80)
    report.append("MODEL EVALUATION REPORT")
    report.append("="*80)
    report.append("")
    
    comparison_df = compare_models(model_results, metric='directional_accuracy')
    report.append("="*80)
    report.append("MODEL RANKING BY DIRECTIONAL ACCURACY (Primary Metric)")
    report.append("="*80)
    report.append(comparison_df.to_string(index=False))
    report.append("")
    
    # Key insights
    report.append("="*80)
    report.append("KEY INSIGHTS FOR TREND PREDICTION")
    report.append("="*80)
    best_model = comparison_df.iloc[0]
    worst_model = comparison_df.iloc[-1]
    report.append(f"Best Model:  {best_model['model']} - {best_model['directional_accuracy']:.2f}% directional accuracy")
    report.append(f"Worst Model: {worst_model['model']} - {worst_model['directional_accuracy']:.2f}% directional accuracy")
    report.append("")
    
    # Trading performance
    report.append("="*80)
    report.append("TRADING PERFORMANCE METRICS")
    report.append("="*80)
    trading_df = comparison_df[['model', 'sharpe_ratio', 'hit_rate', 'profit_factor', 'max_drawdown']].copy()
    report.append(trading_df.to_string(index=False))
    report.append("")
    
    # Numerical accuracy (context only)
    report.append("="*80)
    report.append("NUMERICAL ACCURACY (Context - Not Optimization Target)")
    report.append("="*80)
    accuracy_df = comparison_df[['model', 'rmse', 'mae', 'mape']].copy()
    report.append(accuracy_df.to_string(index=False))
    report.append("")
    
    # Individual model details
    report.append("="*80)
    report.append("DETAILED METRICS BY MODEL")
    report.append("="*80)
    
    for model_name, metrics in model_results.items():
        report.append(f"\n{model_name.upper()}")
        report.append("-"*40)
        
        # Group metrics by importance
        report.append("  Trend Prediction:")
        report.append(f"    directional_accuracy : {metrics['directional_accuracy']:.4f}")
        report.append(f"    r2                   : {metrics['r2']:.4f}")
        
        report.append("  Trading Performance:")
        report.append(f"    sharpe_ratio         : {metrics['sharpe_ratio']:.4f}")
        report.append(f"    hit_rate             : {metrics['hit_rate']:.4f}")
        report.append(f"    profit_factor        : {metrics['profit_factor']:.4f}")
        report.append(f"    max_drawdown         : {metrics['max_drawdown']:.4f}%")
        
        report.append("  Numerical Accuracy:")
        report.append(f"    rmse                 : {metrics['rmse']:.4f}")
        report.append(f"    mae                  : {metrics['mae']:.4f}")
        report.append(f"    mape                 : {metrics['mape']:.4f}%")
    
    report_str = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_str)
        print(f"Evaluation report saved to: {save_path}")
    
    return report_str
