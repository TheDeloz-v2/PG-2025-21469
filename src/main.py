from market_pipeline import run_pipeline as run_market_pipeline
from sentiment_pipeline import run_pipeline as run_sentiment_pipeline
from stock_pipeline import run_pipeline as run_stock_pipeline
from fusion_pipeline import run_pipeline as run_fusion_pipeline


def main():
    """
    Run all pipelines: market, sentiment, stock, and fusion (dataset alignment).
    """
    print("="*70)
    print("INICIANDO MULTIMODAL FINANCIAL PREDICTION AGENT")
    print("="*70)

    # Pipeline 1: Stock Market Analysis
    print("\n--- Pipeline 1: Stock Market Analysis ---")
    market_out = run_market_pipeline(save=True)
    print(f"✓ Stock market dataset guardado en: {market_out}")

    # Pipeline 2: Sentiment Analysis
    print("\n--- Pipeline 2: Sentiment Analysis ---")
    sentiment_out = run_sentiment_pipeline(save=True)
    print(f"✓ Sentiment dataset guardado en: {sentiment_out}")

    # Pipeline 3: Tesla Stock Decomposition
    print("\n--- Pipeline 3: Tesla Stock Decomposition ---")
    stock_out = run_stock_pipeline(save=True, verbose=False)
    print(f"✓ Tesla decomposition dataset guardado en: {stock_out}")

    # Pipeline 4: Fusion (Dataset Alignment)
    print("\n--- Pipeline 4: Dataset Fusion ---")
    datasets, fusion_paths = run_fusion_pipeline(
        save=True, 
        validate=True,
        equalize_dates=True
        )
    print(f"✓ Fusion pipeline completado: {len(datasets)} datasets generados")

    print("\n" + "="*70)
    print("TODOS LOS PIPELINES SE COMPLETARON CORRECTAMENTE ✓")
    print("="*70)
    print(f"\nDatasets generados:")
    print(f"  1. Market (raw): {market_out}")
    print(f"  2. Sentiment (raw): {sentiment_out}")
    print(f"  3. Stock (raw): {stock_out}")
    print(f"\nDatasets alineados (con target consistente):")
    for name, path in fusion_paths.items():
        df = datasets[name]
        shape = df.shape
        date_range = f"{df.index.min().date()} a {df.index.max().date()}"
        print(f"  {name.upper():12}: {shape[0]:4} × {shape[1]:3} | {date_range}")

if __name__ == "__main__":
    main()
