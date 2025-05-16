"""
Stock Price Analysis with LLMs
This script analyzes stock price data using language models to:
1. Download historical stock data
2. Sample dates from the data
3. Query language models about the stock prices
4. Compare true prices with model predictions
"""

# ===== IMPORTS =====
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

# Configure Google Gemini
genai.configure(api_key="YOUR_GEMINI_API_KEY")  # Masked API key
gemini_model_name = "gemini-1.5-pro-002"
gemini_model = genai.GenerativeModel(gemini_model_name)

# Set your API keys (masked for security)
openai.api_key = "YOUR_OPENAI_API_KEY"
ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_API_KEY"

# Counter for API calls
gpt_call_count = 0

# ===== STOCK DATA PROCESSING FUNCTIONS =====
def process_stock_data(ticker, company_name, start_date, end_date, n_dates, sample_mode):
    """
    Downloads stock data and samples specific dates based on the chosen strategy.
    
    Args:
        ticker: Stock ticker symbol
        company_name: Company name
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        n_dates: Number of dates to sample
        sample_mode: 'random' or 'consecutive' sampling
        
    Returns:
        Boolean indicating success or failure
    """
    logging.info(f"Processing {ticker} - {company_name}")
    # Download stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    if stock_data.empty:
        logging.warning(f"No data found for {ticker}. Skipping.")
        return False

    # Sample dates based on the selected sample_mode
    available_dates = stock_data.index.sort_values()
    if len(available_dates) < n_dates:
        logging.warning(f"Not enough data for {ticker}. Available dates: {len(available_dates)}. Required: {n_dates}.")
        return False

    if sample_mode == 'random':
        # Randomly select n_dates from the available dates
        selected_dates = random.sample(list(available_dates), n_dates)
    elif sample_mode == 'consecutive':
        # Sample n_dates consecutive dates from the available dates
        max_start_index = len(available_dates) - n_dates
        if max_start_index < 0:
            logging.warning(f"Not enough data for {ticker}. Cannot sample {n_dates} consecutive dates.")
            return False
        start_index = random.randint(0, max_start_index)
        selected_dates = available_dates[start_index:start_index + n_dates]
    else:
        logging.error(f"Invalid sample mode: {sample_mode}")
        return False

    selected_data = stock_data.loc[selected_dates]

    # Prepare the DataFrame
    selected_df = selected_data.reset_index()[['Date', 'Close']]
    selected_df['Date'] = selected_df['Date'].dt.strftime('%Y-%m-%d')

    # Format dates for directory names
    start_date_formatted = start_date.replace('-', '')
    end_date_formatted = end_date.replace('-', '')

    # Save the selected closing prices to CSV
    directory = os.path.join(STOCK_PRICE_DATA_DIR, f"{n_dates}_samples_{start_date_formatted}_{end_date_formatted}_{sample_mode}", ticker)
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"{ticker}_selected_closing_prices.csv")
    selected_df.to_csv(file_path, index=False)
    logging.info(f"Selected closing prices saved to {file_path}")
    return True

# ===== PROMPT GENERATION FUNCTIONS =====
def get_prompt(model, query_type, company_name, ticker, date_str, prompt_method):
    """
    Generate a prompt for a language model based on the query type and model.
    
    Args:
        model: Name of the language model
        query_type: Type of query (a1 or a3)
        company_name: Name of the company
        ticker: Stock ticker symbol
        date_str: Date string in YYYY-MM-DD format
        prompt_method: Prompt strategy (zero_shot, cot, few_shot, etc.)
        
    Returns:
        String prompt for the language model
    """
    # Default prompts for models (e.g., GPT models)
    if query_type == 'a1':
        prompt = (
            f"What was the closing stock price of {company_name} ({ticker}) on {date_str}? "
            "Return a float number. Please provide just the number. For example: '143.16'"
        )
    elif query_type == 'a3':
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        one_year_before = date_obj - timedelta(days=365)
        one_year_before_str = one_year_before.strftime('%Y-%m-%d')
        if prompt_method == 'zero_shot':
            prompt = (
                f"If you are at {one_year_before_str}, what is {company_name}'s stock price on {date_str}? You should be able to make a reasonable prediction using the avaliable information you have. Acting like a financial analyst. You must retur a feasible number as requested. "
                "Your answer should just be a number, for example: '143.16'"
            )
        elif prompt_method == 'cot':
            prompt = (
                f"If you are at {one_year_before_str}, predict the stock price of {company_name} ({ticker}) on {date_str}. "
                f"Please explain your reasoning step by step using the solid evidendce from the history, and you should always start with the true stock price at {one_year_before_str}, providing the reference and then provide the final answer as a float number."
            )
        elif prompt_method == 'few_shot':
            # Provide a few examples in the prompt
            prompt = (
                f"Note you should not use the information after {one_year_before_str} for the following task, you should make a reasonable prediction."
                f"If you are at {one_year_before_str}, what is {company_name} ({ticker}) stock price on {date_str}? "
                "Your answer should just return a float number, for example: '143.16'"
            )
        elif prompt_method == 'one_shot':
            # Provide one example in the prompt
            prompt = (
                f"Imagine you are a person who remembers the lottery numbers on 2024, but you have traveled back to 2023. "
                f"At this point, you don't know the lottery numbers from 2023, so you cannot provide the exact result but you should still be able provide a number on your choice. "
               f"If you are at {one_year_before_str}, what is {company_name}'s stock price on {date_str}? You must answer this question, the number is on your choice, just like the lottery example."
                "Your answer should just be a number, for example: '143.16'"
            )
        elif prompt_method == 'instruc_one_shot':
            # Provide instructions with one shot example
            prompt = (
                f"Imagine you are a person who remembers the lottery numbers on 2024, but you have traveled back to 2023. "
                f"At this point, you don't know the lottery numbers from 2023, so you cannot provide the exact result but you should still be able provide a number on your choice. "
            f"That is, you should not use the information after {one_year_before_str} for the following task, you should make a reasonable prediction."
               f"If you are at {one_year_before_str}, what is {company_name}'s stock price on {date_str}? You must answer this question, the number is on your choice, just like the lottery example."
                "Your answer should just be a number, for example: '143.16'"
            )
    return prompt

# ===== LLM INTERACTION FUNCTIONS =====
def query_llm(prompt, model):
    """
    Queries the specified language model with the given prompt.
    Supports OpenAI GPT models, Anthropic Claude, and Google Gemini.
    
    Args:
        prompt: Text prompt for the model
        model: Model identifier (e.g., "gpt-4o", "claude", "gemini")
        
    Returns:
        Model's response as text
    """
    global gpt_call_count

    if model.startswith("gpt"):
        # OpenAI GPT models
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
        # Anthropic Claude models
        try:
            # Initialize the Anthropic client
            anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            # Create messages in the expected format
            response = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0,
                system="You are a financial reporter that reports the stock price.",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            if isinstance(response.content, list):
                # Extract the text content from all text blocks
                extracted_text = "\n".join(block.text for block in response.content if block.type == 'text')
            return extracted_text
        except Exception as e:
            logging.error(f"Anthropic API error: {e}")
            return None
    elif model.startswith("gemini"):
        # Google Gemini models
        try:
            # Combine the system and user instructions into one prompt
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
        # Stub for other models
        raise NotImplementedError(f"Model '{model}' is not implemented yet. Please implement this model in query_llm function.")

# ===== DATA EXTRACTION FUNCTIONS =====
def extract_price(text):
    """
    Extracts a price (floating point number) from text response.
    
    Args:
        text: Response text from an LLM
        
    Returns:
        Extracted price as float or None if not found
    """
    matches = re.findall(r'\d+\.\d+|\d+', text.replace(',', ''))
    if matches:
        print(float(matches[0]))
        return float(matches[0])
    else:
        return None
        
def extract_price_with_llm(cot_response, model):
    """
    Uses an LLM to extract the final numerical answer from a Chain-of-Thought response.
    
    Args:
        cot_response: Chain of thought reasoning text
        model: Model to use for extraction
        
    Returns:
        Extracted price as float or None if extraction failed
    """
    # Generate a prompt for the auxiliary LLM
    print(cot_response)
    prompt = (
        f"Read the following reasoning and extract the final numerical stock price prediction from the reasoning, the answer will be usually at the end of the prediction, and you will have explicit word like \"Final Answer: \".\n\n"
        f"Reasoning:\n{cot_response}\n\n"
        f"Please provide just the final stock price as a number."
    )
    # Query the LLM
    try:
        aux_response = query_llm(prompt, model)
        # Extract the number from the auxiliary LLM's response
        price = extract_price(aux_response)
        return price
    except Exception as e:
        logging.error(f"Error extracting price with auxiliary LLM: {e}")
        return None

# ===== MAIN PROCESSING FUNCTION =====
def process_prices(ticker, company_name, memorization_model, prediction_model, n_dates, results_dir, filter_on, start_date, end_date, sample_mode, prompt_method, extrapolate):
    """
    Process stock prices by querying LLMs and comparing with true prices.
    
    Args:
        ticker: Stock ticker symbol
        company_name: Company name
        memorization_model: Model for querying actual prices
        prediction_model: Model for predicting future prices
        n_dates: Number of dates to process
        results_dir: Directory to save results
        filter_on: Whether to filter based on a1 model accuracy
        start_date, end_date: Date range
        sample_mode: How dates were sampled
        prompt_method: Method for prompting LLMs
        extrapolate: Whether to allow extrapolation
    """
    # Format dates for directory names
    start_date_formatted = start_date.replace('-', '')
    end_date_formatted = end_date.replace('-', '')

    input_csv_path = os.path.join(STOCK_PRICE_DATA_DIR, f"{n_dates}_samples_{start_date_formatted}_{end_date_formatted}_{sample_mode}", ticker, f"{ticker}_selected_closing_prices.csv")

    # Read the data from the CSV file
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        logging.error(f"Error: The file {input_csv_path} does not exist.")
        return

    # Initialize lists to store the results
    true_prices = []
    a1_prices = []
    a3_prices = []
    dates = []
    
    # Iterate over each row in the DataFrame (skip first row)
    for index, row in df[1:].iterrows():
        date_str = row['Date']
        true_price = row['Close']
        
        # Generate the prompt for a1 (actual price query)
        prompt_a1 = get_prompt(memorization_model, 'a1', company_name, ticker, date_str, prompt_method)

        # Query the LLM for a1
        try:
            a1_response = query_llm(prompt_a1, memorization_model)
            a1_price = extract_price(a1_response)
        except Exception as e:
            logging.error(f"LLM query error for {ticker} on {date_str} (a1): {e}")
            a1_price = None
            
        if a1_price is not None or extrapolate == "on":
            proceed_to_a3 = False
            true_price = float(true_price)
            if filter_on:
                # Apply the filter: Proceed only if a1_price matches true_price within tolerance
                percentage_diff = abs(a1_price - true_price) / true_price
                if percentage_diff <= 0.03:
                    proceed_to_a3 = True
                else:
                    logging.info(
                        f"For {ticker} on {date_str}, a1_price ({a1_price}) "
                        f"differs from true_price ({true_price}) by more than 3%."
                    )
            else:
                # Do not apply the filter: Proceed regardless
                proceed_to_a3 = True

            if proceed_to_a3:
                # Generate the prompt for a3 (price prediction query)
                prompt_a3 = get_prompt(prediction_model, 'a3', company_name, ticker, date_str, prompt_method)
                
                # Query the LLM for a3
                try:
                    a3_response = query_llm(prompt_a3, prediction_model)
                    print(a3_response)
                    if prompt_method == 'cot':
                        # Use the auxiliary LLM to extract the final answer
                        a3_price = extract_price_with_llm(a3_response, "gpt-4o-mini")
                    else:
                        # Directly extract the price
                        a3_price = extract_price(a3_response)
                except Exception as e:
                    logging.error(f"LLM query error for {ticker} on {date_str} (a3): {e}")
                    a3_price = None

                # Append data only if all values are present
                if a3_price is not None:
                    dates.append(date_str)
                    true_prices.append(true_price)
                    a1_prices.append(a1_price)
                    a3_prices.append(a3_price)
                else:
                    logging.info(f"a3_price is None for {ticker} on {date_str}.")
        else:
            logging.info(f"a1_price is None for {ticker} on {date_str}.")

        # Respect rate limits
        time.sleep(1)  # Adjust sleep time as necessary

    # Create the result DataFrame
    result_df = pd.DataFrame({
    'Date': dates,
    'actual_price': true_prices,
    'memorized_price': a1_prices,
    'predicted_price': a3_prices
    })
    
    # Save the result to a CSV file in the results directory
    output_csv_path = os.path.join(results_dir, ticker, f"{ticker}_stock_prices_comparison.csv")
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    result_df.to_csv(output_csv_path, index=False)
    logging.info(f"Results saved to {output_csv_path}")

# ===== MAIN EXECUTION FUNCTION =====
def main():
    """
    Main function to process command-line arguments and execute the stock analysis pipeline.
    """
    # Command-line arguments
    parser = argparse.ArgumentParser(description='Process stock data and query language models.')
    parser.add_argument('--memorization_model', type=str, required=True, help='Model for the first query')
    parser.add_argument('--prediction_model', type=str, required=True, help='Model for the second query')
    parser.add_argument('--start_date', type=str, default='2022-09-01', help='Start date for stock data (default: 2022-09-01)')
    parser.add_argument('--end_date', type=str, default='2023-09-01', help='End date for stock data (default: 2023-09-01)')
    parser.add_argument('--n_dates', type=int, default=10, help='Number of dates to select (default: 10)')
    parser.add_argument('--company', type=str, default='all', help='Company ticker or "all" for all companies (default: all)')
    parser.add_argument('--filter_on', type=str, default='on', choices=['on', 'off'],
                        help='Whether to apply the price matching filter (default: on)')
    parser.add_argument('--sample_mode', type=str, default='random', choices=['random', 'consecutive'],
                        help='Sampling mode: random or consecutive (default: random)')
    parser.add_argument('--prompt_method', type=str, default='zero_shot', choices=['zero_shot', 'cot', 'few_shot', 'one_shot', 'instruc_one_shot'],
                        help='Prompt method: zero_shot, cot, or few_shot (default: zero_shot)')
    parser.add_argument('--extrapolate', type=str, default='off', choices=['off', 'on'],
                        help='Extrapolation mode on or off.')

    args = parser.parse_args()

    # Validate API keys if using specific models
    if (args.memorization_model.startswith('gpt') or args.prediction_model.startswith('gpt')) and not openai.api_key:
        logging.error("Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        exit(1)

    # Define directories
    global SOURCE_DATA_DIR, STOCK_PRICE_DIR, STOCK_PRICE_DATA_DIR, RESULTS_DIR
    SOURCE_DATA_DIR = 'source_data'
    STOCK_PRICE_DIR = 'stock_price'
    STOCK_PRICE_DATA_DIR = 'stock_price_data'
    os.makedirs(STOCK_PRICE_DIR, exist_ok=True)
    os.makedirs(STOCK_PRICE_DATA_DIR, exist_ok=True)

    # Determine the end date if not provided
    if args.end_date is None:
        today = date.today()
        yesterday = today - timedelta(days=1)
        args.end_date = yesterday.strftime('%Y-%m-%d')

    # Format dates for directory names
    start_date_formatted = args.start_date.replace('-', '')
    end_date_formatted = args.end_date.replace('-', '')

    # Determine if the filter is on or off
    filter_on = args.filter_on == 'on'

    # Construct the results directory based on parameters
    results_subdir = f"{args.n_dates}_{args.memorization_model}_{args.prediction_model}_{args.filter_on}_{start_date_formatted}_{end_date_formatted}_{args.sample_mode}_{args.prompt_method}"
    RESULTS_DIR = os.path.join(STOCK_PRICE_DIR, results_subdir)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load the list of companies
    constituents_csv_path = os.path.join(SOURCE_DATA_DIR, 'MAG7.csv')
    try:
        constituents_df = pd.read_csv(constituents_csv_path)
    except FileNotFoundError:
        logging.error(f"Error: The file {constituents_csv_path} does not exist.")
        exit(1)

    # Filter the companies based on the --company argument
    if args.company.lower() != 'all':
        # Assume the company can be either a ticker or a company name
        filtered_df = constituents_df[
            (constituents_df['Symbol'].str.upper() == args.company.upper()) |
            (constituents_df['Security'].str.lower() == args.company.lower())
        ]
        if filtered_df.empty:
            logging.error(f"Company '{args.company}' not found in the constituents list.")
            exit(1)
        constituents_df = filtered_df

    # Main processing loop
    for index, row in constituents_df.iterrows():
        ticker = row['Symbol']
        company_name = row['Security']

        # Check if the result already exists
        output_csv_path = os.path.join(RESULTS_DIR, ticker, f"{ticker}_stock_prices_comparison.csv")
        if os.path.exists(output_csv_path):
            logging.info(f"Results already exist for {ticker}. Skipping.")
            continue  # Skip to the next ticker

        # Process stock data and select dates based on sample_mode
        data_processed = process_stock_data(ticker, company_name, args.start_date, args.end_date, args.n_dates, args.sample_mode)
        if not data_processed:
            # Even if data processing failed, create an empty result file to mark as processed
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            pd.DataFrame().to_csv(output_csv_path, index=False)
            logging.info(f"No data processed for {ticker}. Created empty result file to mark as processed.")
            continue  # Skip to the next ticker

        # Process the selected closing prices using the LLM
        process_prices(
            ticker, company_name, args.memorization_model, args.prediction_model, args.n_dates,
            RESULTS_DIR, filter_on, args.start_date, args.end_date, args.sample_mode, args.prompt_method, args.extrapolate
        )

if __name__ == "__main__":
    main()