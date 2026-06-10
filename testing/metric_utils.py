import pandas as pd

def get_error_summary(df):
    """Returns a clean DataFrame summarizing Baseline vs Persistence Absolute Error metric."""
    metrics = ['AE']
    all_cols = metrics + [f'Persistence_{m}' for m in metrics]
    
    # 1. Get standard descriptive stats
    summary = df[all_cols].agg(['mean', 'max', 'min', 'std'])
    summary.index = ['Mean', 'Maximum', 'Minimum', 'Standard Deviation']
    
    # 2. Get the specific quantiles
    quantiles = [0.80, 0.85, 0.90, 0.92, 0.95, 0.99]
    quant_df = df[all_cols].quantile(quantiles)
    quant_df.index = [f"{int(q*100)}% Quantile" for q in quantiles]
    
    # 3. Combine them
    full_summary = pd.concat([summary, quant_df])
    
    # 4. Optional: Format columns side-by-side for easier reading
    formatted_data = {}
    for m in metrics:
        display_name = "R²" if m == "R2" else m
        formatted_data[f'{display_name} (Base / Persist)'] = (
            full_summary[m].map('{:.6f}'.format) + " / " + 
            full_summary[f'Persistence_{m}'].map('{:.6f}'.format)
        )

    # Naming the first index column and the title
    df.index.name = 'Metrics'
    df.style.set_caption("Metrics for Absolute Error (AE)")
    
    return pd.DataFrame(formatted_data)
