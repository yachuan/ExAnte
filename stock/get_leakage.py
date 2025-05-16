import os
import pandas as pd
from datetime import datetime


FOLDER = "Self-Ver-In-Conv/"  # root folder
TICKERS = ['GOOGL', 'AAPL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META']
MODELS = ['gpt-4o', 'claude', 'gemini']
PROMPT_METHODS = ['cot', 'Instruction-Based', 'one_shot', 'zero_shot', 'self_verification_hint', 'self_verification_no_hint']
threshold = 0.03

def compute_leakage_rates():
    leakage_records = []

    for ticker in TICKERS:
        for model in MODELS:
            file_path = os.path.join(FOLDER, f"{model}_{ticker}.csv")
            if not os.path.exists(file_path):
                print(f"⚠️ Missing file: {file_path}, skipping.")
                continue

            df = pd.read_csv(file_path)
            if 'actual_price' not in df.columns:
                print(f"⚠️ No 'actual_price' in {file_path}, skipping.")
                continue

            for prompt in PROMPT_METHODS:
                if prompt not in df.columns:
                    continue
                try:
                    preds = pd.to_numeric(df[prompt], errors='coerce')
                    actuals = pd.to_numeric(df['actual_price'], errors='coerce')
                    valid_mask = preds.notna() & actuals.notna()

                    preds = preds[valid_mask]
                    actuals = actuals[valid_mask]

                    leakage_mask = (abs(preds - actuals) / actuals) <= threshold
                    leakage_rate = leakage_mask.mean()  # proportion of leakage

                    leakage_records.append({
                        "model": model,
                        "prompt_method": prompt,
                        "ticker": ticker,
                        "leakage_rate": leakage_rate
                    })
                except Exception as e:
                    print(f"⚠️ Error processing {file_path} / {prompt}: {e}")

    df_leakage = pd.DataFrame(leakage_records)

    # Aggregate across tickers: mean leakage rate per model + prompt
    agg_leakage = df_leakage.groupby(["model", "prompt_method"])["leakage_rate"].mean().unstack()
    agg_leakage.to_csv(f'leakage_rates_{threshold}.csv')
    print(f'✅ Saved leakage rates to leakage_rates_{threshold}.csv')


if __name__ == "__main__":
    compute_leakage_rates()
