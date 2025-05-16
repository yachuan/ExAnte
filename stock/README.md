# Stock Price Prediction Analysis Pipeline README

This folder contains tools for analyzing stock price predictions made by language models (LLMs) compared to human analyst forecasts.

## 0. Data Preparation

### Analyst Data
- Put analyst rating history files into `analyst/` directory
- Files should follow naming convention: `NASDAQ-{TICKER}-Analyst-Rating-History.csv`
- Available for: AAPL, AMZN, GOOGL, META, MSFT, TSLA, NVDA

## 1. Generate Model Predictions

### Run prediction pipeline:
```bash
python exante-main.py --memorization_model MODEL_NAME --prediction_model MODEL_NAME 
  --prompt_method METHOD --sample_mode random --n_dates 10
```

- Supported models: gpt-4o, claude, gemini
- Prompt methods: zero_shot, cot, few_shot, one_shot, instruc_one_shot
- Results saved to: `stock_price/{n_dates}_{memorization_model}_{prediction_model}_{filter}_{dates}_{sample}_{method}/{ticker}/{ticker}_stock_prices_comparison.csv`

## 2. Run Self-Verification Tests

### Independent verification:
```bash
python self-verification-independent.py --ticker all --memorization_model MODEL_NAME
```
- Results saved to: `Self-Ver-Independent/{model}/{model}_{ticker}.csv`

### In-conversation verification:
```bash
python self-verification-in-conversation.py --ticker all --memorization_model MODEL_NAME
```
- Results saved to: `Self-Ver-In-Conv/{model}/{model}_{ticker}.csv`

## 3. Merge Results & Calculate Metrics

### Merge prediction results:
```bash
jupyter notebook merge-result.ipynb
```
- Combines different prompting strategy results into single files
- Updates files in: `Self-Ver-In-Conv/`

### Calculate Leakage Rate:
```bash
python get_leakage.py
```
- Calculates the rate at which model predictions are suspiciously close to actual prices
- A prediction is considered "leakage" if within 3% of actual price
- Results saved to: leakage_rates_0.03.csv

### Calculate accuracy against analyst targets:
```bash
python get_accuracy.py
```
- Compares model predictions with analyst price targets
- Results saved to:
  - `analyst_vs_model_absolute_errors.csv` (detailed)
  - `aggregated_absolute_errors.csv` (summary)

