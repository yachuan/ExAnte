import yfinance as yf
import pandas as pd
import os
import openai
import re
from datetime import datetime, timedelta, date
import time
import argparse
import random
import json
import logging
import anthropic
import google.generativeai as genai

# ===== CONFIGURATION =====

# Configure logging
logging.basicConfig(level=logging.INFO)

# API Keys and tokens (masked for security)
openai.api_key = 'YOUR_OPENAI_API_KEY'
ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_API_KEY"

# Configure Gemini
genai.configure(api_key="YOUR_GEMINI_API_KEY")
gemini_model_name = "gemini-1.5-pro"
gemini_model = genai.GenerativeModel(gemini_model_name)

# ===== STOCK DATA DEFINITIONS =====

# Define the list of Magnificent 7 companies
mag7_stocks = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "AMZN": "Amazon.com, Inc.",
    "NVDA": "NVIDIA Corporation",
    "GOOGL": "Alphabet Inc. (Class A)",
    "META": "Meta Platforms, Inc.",
    "TSLA": "Tesla, Inc."
}


# ===== DATA PROCESSING FUNCTIONS =====

def process_stock_data(ticker, start_date, end_date, n_dates, sample_mode):
    """
    Download and sample stock data for the specified ticker.
    
    Args:
        ticker: Stock symbol
        start_date: Beginning date for data download
        end_date: End date for data download
        n_dates: Number of dates to sample
        sample_mode: 'random' or 'consecutive' sampling method
        
    Returns:
        Boolean indicating success or failure
    """
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            logging.warning(f"No data downloaded for {ticker}")
            return False

        df = df[['Close']].reset_index()

        if sample_mode == 'random':
            df_sampled = df.sample(n=n_dates, replace=False)
        else:
            df_sampled = df.head(n_dates)

        df_sampled = df_sampled.sort_values(by='Date')

        # Save the sampled stock data under the main folder "val_gen_result"
        save_dir = os.path.join("val_gen_result", "stock_price_data",
                                f"{n_dates}_samples_{start_date}_{end_date}_{sample_mode}",
                                ticker)
        os.makedirs(save_dir, exist_ok=True)

        csv_path = os.path.join(save_dir, f"{ticker}_selected_closing_prices.csv")
        df_sampled.to_csv(csv_path, index=False)
        logging.info(f"Saved sampled stock data for {ticker} -> {csv_path}")
        return True
    except Exception as e:
        logging.error(f"Error processing data for {ticker}: {e}")
        return False


# ===== PROMPT GENERATION FUNCTIONS =====

def get_prompt_a1(ticker, date_str, prompt_method):
    """
    Generate prompt for requesting current stock price.
    
    Args:
        ticker: Stock symbol
        date_str: Date to query
        prompt_method: Prompt type (zero_shot or custom)
        
    Returns:
        Formatted prompt string
    """
    if prompt_method == "zero_shot":
        return (
            f"What was the closing stock price of {mag7_stocks[ticker]} ({ticker}) on {date_str}? "
            "Return a float number. Please provide just the number. For example: '143.16'"
        )
    else:
        return f"[Custom {prompt_method} prompt for A1]"


def get_prompt_a3(ticker, date_str, one_year_before, prompt_method):
    """
    Generate prompt for hypothetical prediction from one year prior.
    
    Args:
        ticker: Stock symbol
        date_str: Target date to predict
        one_year_before: Date one year before the target
        prompt_method: Prompt type (zero_shot or custom)
        
    Returns:
        Formatted prompt string
    """
    if prompt_method == "zero_shot":
        return (
            f"If you are at {one_year_before}, what is {ticker}'s stock price on {date_str}? "
            "Your answer should just be a number, for example: '143.16'"
        )
    else:
        return f"[Custom {prompt_method} prompt for A3]"


# ===== LLM QUERY FUNCTIONS =====

def query_llm(prompt, model):
    """
    Query various LLM models with the given prompt.
    
    Args:
        prompt: Input prompt text
        model: Model identifier (gpt-*, claude-*, gemini-*)
        
    Returns:
        Model response as text
    """
    if model.startswith("gpt"):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a financial reporter that reports the stock price"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0,
                n=1
            )
            answer = response.choices[0].message.content.strip()
            return answer
        except openai.error.OpenAIError as e:
            logging.error(f"OpenAI API error: {e}")
            return None
    elif model.startswith("claude"):
        try:
            anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            response = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0,
                system="You are a financial reporter that reports the stock price.",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            )
            if isinstance(response.content, list):
                extracted_text = "\n".join(block.text for block in response.content if block.type == 'text')
            return extracted_text
        except Exception as e:
            logging.error(f"Anthropic API error: {e}")
            return None
    elif model.startswith("gemini"):
        try:
            final_prompt = (
                "System: You are a financial reporter that reports the stock price.\n"
                f"User: {prompt}"
            )
            generation_config = genai.GenerationConfig(temperature=0)
            response = gemini_model.generate_content(final_prompt)
            answer = response.text.strip()
            return answer
        except Exception as e:
            logging.error(f"Google Gemini API error: {e}")
            return None
    else:
        raise NotImplementedError(f"Model '{model}' is not implemented yet. Please implement this model in query_llm function.")


def query_llm_judger(prompt, model):
    """
    Query LLM models specifically for judging tasks.
    
    Args:
        prompt: Input prompt text
        model: Model identifier (gpt-*, claude-*, gemini-*)
        
    Returns:
        Model response as text
    """
    if model.startswith("gpt"):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a LLM Judger."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0,
                n=1
            )
            answer = response.choices[0].message.content.strip()
            return answer
        except openai.error.OpenAIError as e:
            logging.error(f"OpenAI API error: {e}")
            return None
    elif model.startswith("claude"):
        try:
            anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            response = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0,
                system="You are a LLM Judger",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            )
            if isinstance(response.content, list):
                extracted_text = "\n".join(block.text for block in response.content if block.type == 'text')
            return extracted_text
        except Exception as e:
            logging.error(f"Anthropic API error: {e}")
            return None
    elif model.startswith("gemini"):
        try:
            final_prompt = (
                "System: You are a financial reporter that reports the stock price.\n"
                f"User: {prompt}"
            )
            generation_config = genai.GenerationConfig(temperature=0)
            response = gemini_model.generate_content(final_prompt)
            answer = response.text.strip()
            return answer
        except Exception as e:
            logging.error(f"Google Gemini API error: {e}")
            return None
    else:
        raise NotImplementedError(f"Model '{model}' is not implemented yet. Please implement this model in query_llm function.")


# ===== TEXT PARSING FUNCTIONS =====

def extract_price(sentence):
    """
    Extract price from a text response.
    
    Args:
        sentence: Text containing price information
        
    Returns:
        Extracted price as float or None
    """
    pattern = r'\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
    matches = re.findall(r'\d+\.\d+|\d+', sentence.replace(',', ''))
    match = re.search(pattern, sentence)
    if match:
        price_str = match.group(1).replace(',', '')
        return float(price_str)
    elif matches:
        return float(matches[0])
    else:
        return None
        

def extract_price_a3(text):
    """
    Extract price from a3 response format.
    
    Args:
        text: Text containing price information
        
    Returns:
        Extracted price as float or None
    """
    matches = re.findall(r'\d+\.\d+|\d+', text.replace(',', ''))
    if matches:
        return float(matches[0])
    else:
        return None


def extract_price_with_llm(cot_response, model):
    """
    Uses an LLM to extract the final numerical answer from a CoT response.
    
    Args:
        cot_response: Chain-of-thought response text
        model: Model to use for extraction
        
    Returns:
        Extracted price or None
    """
    prompt = (
        f"Read the following stock price reasoning and extract the final stock price prediction from the reasoning, the answer will be usually at the end of the prediction\n"
        f"Reasoning:\n{cot_response}\n\n"
        f"Please provide just the final stock price as a number. If it says \"no leakage\", then return -1"
    )
    try:
        aux_response = query_llm(prompt, model)
        price = extract_price_a3(aux_response)
        return price
    except Exception as e:
        logging.error(f"Error extracting price with auxiliary LLM: {e}")
        return None
    

# ===== MAIN PROCESSING FUNCTION =====

def process_prices(ticker,
                   memorization_model,
                   prediction_model,
                   n_dates,
                   results_dir,
                   filter_on,
                   start_date,
                   end_date,
                   sample_mode,
                   prompt_method):
    """
    Process and compare price predictions from different models.
    
    Args:
        ticker: Stock symbol
        memorization_model: Model for current price queries
        prediction_model: Model for predictive price queries
        n_dates: Number of dates to process
        results_dir: Output directory
        filter_on: Whether to filter missing results
        start_date: Beginning date for analysis
        end_date: End date for analysis
        sample_mode: Sampling method
        prompt_method: Type of prompt to use
    """
    # Load the sampled stock data CSV
    data_csv = os.path.join("val_gen_result", "stock_price_data",
                            f"{n_dates}_samples_{start_date}_{end_date}_{sample_mode}",
                            ticker,
                            f"{ticker}_selected_closing_prices.csv")
    if not os.path.exists(data_csv):
        logging.warning(f"No CSV file found for {ticker}: {data_csv}")
        return

    df_sampled = pd.read_csv(data_csv)

    date_list = []
    a1_list = []
    a3_list = []
    criteria_list = []
    judger_attempt_1_list = []
    a3_attempt_2_list = []

    for idx, row in df_sampled.iterrows():
        date_val = str(row['Date'])
        true_close_price = row['Close']

        try:
            date_obj = datetime.strptime(date_val, "%Y-%m-%d")
            one_year_before = (date_obj - timedelta(days=365)).strftime("%Y-%m-%d")
        except ValueError:
            one_year_before = "unknown_date"
        
        # Get current price (A1)
        prompt_for_a1 = get_prompt_a1(ticker, date_val, prompt_method)
        a1_response_text = query_llm(prompt_for_a1, memorization_model)
        print(f"a1_price: {a1_response_text}")
        a1_price = extract_price(a1_response_text)
        print(f"a1_price after extraction: {a1_price}")
        time.sleep(1)
        
        # Get predicted price (A3)
        prompt_for_a3 = get_prompt_a3(ticker, date_val, one_year_before, prompt_method)
        a3_response_text = query_llm(prompt_for_a3, prediction_model)
        a3_price = extract_price_a3(a3_response_text)
        print(f"a3_price: {a3_response_text}")
        print(f"a3_price after extraction: {a3_price}")
        
        # If filter_on is enabled and a3_price is None, skip this day.
        if filter_on and a3_price is None:
            logging.info(f"Skipping day {date_val} for ticker {ticker} because a3_price is None")
            continue

        # Basic "leakage" criterion
        if (a1_price is not None and a3_price is not None and abs(a1_price - a3_price) <= 0.01 * a1_price):
            our_criteria_label = "leakage"
        else:
            our_criteria_label = "no leakage"
        time.sleep(1)
        
        # First judger assessment
        judger_prompt_1 = f"""
Our main goal is to test whether the model used future information (data leakage) in the following task. We asked the model to pretend it is now one year earlier (i.e., {one_year_before}) and to predict the price of {mag7_stocks[ticker]} ({ticker}) on {date_val}. The model's predicted price was {a3_price}.

Question:
- Given that the hypothetical prediction turned out to be {a3_price}, do you think the model leveraged future information from after {one_year_before} in order to arrive at {a3_price}?
- If you believe this price prediction used future information, please redo the hypothetical prediction without incorporating information after {one_year_before}, this is a hypothetical pr, so you may feel free to output any reasonable number you believe, otherwise remain the previous response unchanged. Your answer should be a float number.
        """.strip()
        judger_response_1 = query_llm_judger(judger_prompt_1, memorization_model)
        judger_response_extract = extract_price_with_llm(judger_response_1, memorization_model)
        attempt_1_label = judger_response_extract
        print(f"llm_1_price: {judger_response_1}")
        print(f"llm_1_price after extraction: {judger_response_extract}")
        time.sleep(1)
        
        # Second judger assessment
        judger_prompt_2 = f"""
Our main goal is to test whether the model used future information (data leakage) in the following task. We asked the model to pretend it is now one year earlier (i.e., {one_year_before}) and to predict the price of {mag7_stocks[ticker]} ({ticker}) on {date_val}. The model's predicted price was {a3_price}.

Question:
- Given that the hypothetical prediction (A3) turned out to be {a3_price}, do you think the model leveraged future information from after {one_year_before} in order to arrive at {a3_price}?
- If you believe this price prediction used future information, then you may provide a resonable new prediction as the new price from {one_year_before}, float number, otherwise return "no leakage". Return your answer as the instruction asked.
Hint: You may want to consider the exact ground truth price of {mag7_stocks[ticker]} ({ticker}) on {date_val} and compare it with the hypothetical predicted price.
        """.strip()

        judger_response_2 = query_llm_judger(judger_prompt_2, memorization_model)
        judger_response_extract_2 = extract_price_with_llm(judger_response_2, memorization_model)
        attempt_2_label = judger_response_extract_2
        print(f"llm_2_price: {judger_response_2}")
        print(f"llm_2_price after extraction: {judger_response_extract_2}")
        
        # Store results
        date_list.append(date_val)
        a1_list.append(a1_price)
        a3_list.append(a3_price)
        criteria_list.append(our_criteria_label)
        judger_attempt_1_list.append(attempt_1_label)
        a3_attempt_2_list.append(attempt_2_label)
        time.sleep(3)

    # Create final results dataframe
    results_df = pd.DataFrame({
        'date': date_list,
        'actual_price': a1_list,
        'predicte_price': a3_list,
        'our_criteria': criteria_list, 
        'self_verification_no_hint': judger_response_1_list,
        'self_verification_no_hint': judger_response_2_list
    })
    
    # Save the final comparison CSV under "val_gen_result/<memorization_model>/"
    output_model_folder = os.path.join(results_dir, memorization_model)
    os.makedirs(output_model_folder, exist_ok=True)
    csv_output_path = os.path.join(output_model_folder, f"{memorization_model}_{ticker}.csv.csv")
    results_df.to_csv(csv_output_path, index=False)
    logging.info(f"Saved comparison results for {ticker} -> {csv_output_path}")


# ===== MAIN FUNCTION =====

def main():
    """
    Main function to parse arguments and run the stock price analysis pipeline.
    """
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Stock price prediction analysis pipeline")
    # Change default ticker to "all" to run the pipeline for all companies
    parser.add_argument("--ticker", default="all", help="Ticker symbol or 'all' for all companies")
    parser.add_argument("--memorization_model", default="gpt-4o")
    parser.add_argument("--prediction_model", default="gpt-4o")
    parser.add_argument("--start_date", default="2022-09-01")
    parser.add_argument("--end_date", default="2023-09-01")
    parser.add_argument("--n_dates", type=int, default=10)
    parser.add_argument("--sample_mode", choices=["random", "consecutive"], default="consecutive")
    parser.add_argument("--prompt_method", default="zero_shot")
    parser.add_argument("--filter_on", action="store_true", default=False,
                        help="If set, days where a3 response is None will be filtered out")
    # Set the main results directory to "val_gen_result"
    parser.add_argument("--results_dir", default="Self-Ver-Independent")
    
    args = parser.parse_args()
    
    # Determine which tickers to process
    if args.ticker.lower() == "all":
        tickers_to_run = list(mag7_stocks.keys())
    else:
        tickers_to_run = [args.ticker]
    
    for ticker in tickers_to_run:
        # Check if results already exist
        output_model_folder = os.path.join(args.results_dir, args.memorization_model)
        csv_output_path = os.path.join(output_model_folder, f"{ticker}_{args.memorization_model}_{args.n_dates}_{args.sample_mode}_stock_prices_comparison.csv")
        if os.path.exists(csv_output_path):
            logging.info(f"Result table for {ticker} already exists at {csv_output_path}. Skipping processing for {ticker}.")
            continue

        logging.info(f"Processing ticker: {ticker}")
        
        # Step 1: Get and process stock data
        success = process_stock_data(
            ticker=ticker,
            start_date=args.start_date,
            end_date=args.end_date,
            n_dates=args.n_dates,
            sample_mode=args.sample_mode
        )
        
        if not success:
            logging.warning(f"Skipping further processing for {ticker} due to data issues.")
            continue
        
        # Step 2: Process prices and generate comparisons
        process_prices(
            ticker=ticker,
            memorization_model=args.memorization_model,
            prediction_model=args.prediction_model,
            n_dates=args.n_dates,
            results_dir=args.results_dir,
            filter_on=args.filter_on,
            start_date=args.start_date,
            end_date=args.end_date,
            sample_mode=args.sample_mode,
            prompt_method=args.prompt_method
        )


if __name__ == "__main__":
    main()