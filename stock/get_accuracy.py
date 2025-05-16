import os
import pandas as pd
from datetime import datetime

# Configuration
FOLDER = "Self-Ver-In-Conv/"  # root folder
ANALYST_FOLDER = os.path.join("./", "analyst")
TICKERS = ['GOOGL', 'AAPL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META']
MODELS = ['gpt-4o', 'claude', 'gemini']
PROMPT_METHODS = ['cot', 'Instruction-Based', 'one_shot', 'zero_shot', 'self_verification_hint', 'self_verification_no_hint']

def parse_price_target(pt_str):
    try:
        if '➝' in pt_str:
            _, target = pt_str.split('➝')
        else:
            target = pt_str
        return float(target.replace('$', '').replace(',', '').strip())
    except:
        return None

def find_nearest_date(df, target_date):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    target_date = pd.to_datetime(target_date)
    available_dates = df[df['date'] <= target_date]
    if available_dates.empty:
        return None
    return available_dates.sort_values(by='date', ascending=False).iloc[0]

def main():
    results = []

    for ticker in TICKERS:
        analyst_file = os.path.join(ANALYST_FOLDER, f"NASDAQ-{ticker}-Analyst-Rating-History.csv")
        if not os.path.exists(analyst_file):
            print(f"⚠️ Missing analyst file for {ticker}, skipping.")
            continue

        analyst_df = pd.read_csv(analyst_file)
        analyst_df['DATE'] = pd.to_datetime(analyst_df['DATE'], errors='coerce')

        for _, row in analyst_df.iterrows():
            date = row['DATE']
            price_target_raw = str(row.get('PRICE TARGET', ''))
            price_target = parse_price_target(price_target_raw)
            if pd.isna(date) or price_target is None:
                continue

            for model in MODELS:
                model_file = os.path.join(FOLDER, f"{model}_{ticker}.csv")
                if not os.path.exists(model_file):
                    print(f"⚠️ Missing model file: {model_file}, skipping.")
                    continue

                model_df = pd.read_csv(model_file)
                model_row = find_nearest_date(model_df, date)
                if model_row is None:
                    continue

                for prompt in PROMPT_METHODS:
                    if prompt not in model_row or pd.isna(model_row[prompt]):
                        continue
                    try:
                        model_pred = float(model_row[prompt])
                        abs_error = abs(model_pred - price_target)
                        results.append({
                            "date": date.date(),
                            "ticker": ticker,
                            "model": model,
                            "prompt_method": prompt,
                            "analyst_target": price_target,
                            "model_prediction": model_pred,
                            "absolute_error": abs_error
                        })
                    except:
                        continue

    results_df = pd.DataFrame(results)
    results_df.to_csv("analyst_vs_model_absolute_errors.csv", index=False)
    print("✅ Saved detailed results to analyst_vs_model_absolute_errors.csv")

    # Aggregate: mean absolute error per model x prompt method
    agg_df = results_df.groupby(['model', 'prompt_method'])['absolute_error'].mean().unstack()
    agg_df.to_csv("aggregated_absolute_errors.csv")
    print("✅ Saved aggregated results to aggregated_absolute_errors.csv")

if __name__ == "__main__":
    main()
