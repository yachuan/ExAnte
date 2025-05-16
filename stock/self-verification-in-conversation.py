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
from llamaapi import LlamaAPI
import logging
import anthropic
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor

###################################
# Configuration and API Settings  #
###################################

# Configure Gemini
genai.configure(api_key="YOUR-OWN-API-KEY")
gemini_model_name = "gemini-1.5-pro"
gemini_model = genai.GenerativeModel(gemini_model_name)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define your two GPT API keys for rotation.
gpt_api_keys = [
    'YOUR_OPENAI_API_KEY',
    'YOUR_OPENAI_API_KEY'
]
gpt_call_count = 0  # Global counter to track GPT calls

# API Keys and tokens for other services
openai.api_key = gpt_api_keys[0]  # Default starting key
ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_API_KEY"
llama = None  # Will initialize if needed

# Define the list of companies
mag7_stocks = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "AMZN": "Amazon.com, Inc.",
    "NVDA": "NVIDIA Corporation",
    "GOOGL": "Alphabet Inc. (Class A)",
    "META": "Meta Platforms, Inc.",
    "TSLA": "Tesla, Inc."
}

###################################
# LLM Interaction Functions       #
###################################

def query_llm_context(conversation, model, system_message=None):
    """
    Query an LLM using an existing conversation context.
    The conversation should be a list of dictionaries with keys "role" and "content".
    Optionally, you can specify a system_message.
    """
    if system_message:
        system_msg = {"role": "system", "content": system_message}
        messages = [system_msg] + conversation
    else:
        messages = conversation

    if model.startswith("gpt"):
        # Rotate GPT API keys every 30 calls
        global gpt_call_count
        key_index = (gpt_call_count // 30) % len(gpt_api_keys)
        openai.api_key = gpt_api_keys[key_index]
        gpt_call_count += 1

        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=1000,
                temperature=0,
                n=1
            )
            answer = response.choices[0].message.content.strip()
            print(answer)  
            return answer
        except openai.error.OpenAIError as e:
            logging.error(f"OpenAI API error: {e}")
            return None
    elif model.startswith("llama"):
        global llama
        if llama is None:
            llama = LlamaAPI(llama_api_token)
        api_request_json = {
            "model": model,
            "messages": conversation,
            "stream": False
        }
        try:
            response = llama.run(api_request_json)
            response_json = response.json()
            answer = response_json['choices'][0]['message']['content']
            return answer.strip()
        except Exception as e:
            logging.error(f"Llama API error: {e}")
            return None
    elif model.startswith("claude"):
        try:
            anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            response = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0,
                system=system_message if system_message else "You are a financial reporter that reports the stock price.",
                messages=[
                    {
                        "role": "user",
                        "content": "\n".join(msg["content"] for msg in conversation if msg["role"] == "user")
                    }
                ]
            )
            if isinstance(response.content, list):
                extracted_text = "\n".join(block.text for block in response.content if block.type == 'text')
            else:
                extracted_text = response.content
            return extracted_text
        except Exception as e:
            logging.error(f"Anthropic API error: {e}")
            return None
    elif model.startswith("gemini"):
        try:
            # If there is a conversation history, use the chat API.
            if conversation:
                if system_message and conversation[0]["content"].strip() == "":
                    conversation[0]["content"] = f"*{system_message}*\n{conversation[0]['content']}"
                history = [
                    {"role": msg["role"], "parts": msg["content"]}
                    for msg in conversation if msg["content"].strip()
                ]
                if not history:
                    history = [{"role": "user", "parts": "Please provide your answer."}]
                chat = gemini_model.start_chat(history=history)
                default_message = "Proceed."
                answer_response = chat.send_message(default_message)
                answer = answer_response.text.strip() if answer_response.text else ""
                print(answer)
                return answer
            else:
                final_prompt = ""
                if system_message:
                    final_prompt += f"System: {system_message}\n"
                for msg in conversation:
                    role = msg["role"].capitalize()
                    final_prompt += f"{role}: {msg['content']}\n"
                if not final_prompt.strip():
                    final_prompt = "Please provide your answer."
                response = gemini_model.generate_content(final_prompt)
                answer = response.text.strip() if response.text else ""
                return answer
        except Exception as e:
            logging.error(f"Google Gemini API error: {e}")
            return ""
    else:
        raise NotImplementedError(f"Model '{model}' is not implemented yet in query_llm_context.")

###################################
# Data Extraction Functions       #
###################################

def extract_price(sentence):
    """
    Regex-based extraction for a stock price from a string.
    """
    pattern = r'\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
    match = re.search(pattern, sentence)
    if match:
        price_str = match.group(1).replace(',', '')
        try:
            return float(price_str)
        except ValueError:
            return None
    matches = re.findall(r'\d+\.\d+|\d+', sentence.replace(',', ''))
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            return None
    else:
        return None

def extract_price_a3(text):
    """
    For A3, we simply use the same regex-based extraction.
    """
    return extract_price(text)

def llm_extract_price(response_text, model):
    """
    Use an LLM judger to extract the final stock price from a given response.
    The LLM is instructed to return either a float (as a number) or the exact phrase "no leakage".
    """
    extraction_prompt = (
        "Extract the final stock price prediction from the following content. It usually at the end of context."
        "Only return the price as a float number (for example, 143.16). If the content includes meaning of \"no leakage being detected\", then output 1"
        f"Response:\n{response_text}"
    )
    extraction_conversation = [{"role": "user", "content": extraction_prompt}]
    extracted_text = query_llm_context(extraction_conversation, model, system_message="You are a LLM Judger for extracting numerical prices.")
    if extracted_text is None:
        return None
    extracted_text = extracted_text.strip()
    if extracted_text.lower() == "no leakage":
        return "no leakage"
    try:
        return float(extracted_text)
    except Exception as e:
        logging.error(f"Failed to extract price using LLM extraction from: {extracted_text}")
        return extracted_text

###################################
# Data Processing Functions       #
###################################

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

    data_csv = os.path.join("Gen_Val_Result_2", "stock_price_data",
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
    judger_response_1_list = []
    judger_response_2_list = []

    iteration_counter = 0  # <-- Initialize a counter

    for idx, row in df_sampled.iterrows():
        iteration_counter += 1  # Increment counter each iteration
        date_val = str(row['Date'])
        true_close_price = row['Close']

        try:
            date_obj = datetime.strptime(date_val, "%Y-%m-%d")
            one_year_before = (date_obj - timedelta(days=365)).strftime("%Y-%m-%d")
        except ValueError:
            one_year_before = "unknown_date"

        # --- Memorization Query ---
        prompt_for_a1 = get_prompt_a1(ticker, date_val, prompt_method)
        print(prompt_for_a1)
        conversation = [{"role": "user", "content": prompt_for_a1}]
        a1_response_text = query_llm_context(
            conversation,
            memorization_model,
            system_message="You are a financial reporter that reports the stock price."
        )
        conversation.append({"role": "assistant", "content": a1_response_text})
        a1_price = extract_price(a1_response_text)
        logging.info(f"[{ticker} {date_val}] A1 price extracted: {a1_price}")
        
        # --- Filter for Memorized Price ---
        if filter_on:
            if not isinstance(a1_price, float) or abs(a1_price - true_close_price) > 0.01 * true_close_price:
                logging.info(f"Skipping day {date_val} for ticker {ticker} because A1 price {a1_price} does not match true stock price {true_close_price}")
                continue
        time.sleep(0.5)
        # --- Prediction Price Query ---
        conversation = []
        prompt_for_a3 = get_prompt_a3(ticker, date_val, one_year_before, prompt_method)
        conversation.append({"role": "user", "content": prompt_for_a3})
        a3_response_text = query_llm_context(
            conversation,
            prediction_model,
            system_message="You are a financial reporter that reports the stock price."
        )
        print(f"a3_response_text: {a3_response_text}")
        conversation.append({"role": "assistant", "content": a3_response_text})
        a3_price = extract_price_a3(a3_response_text)
        logging.info(f"[{ticker} {date_val}] A3 price extracted: {a3_price}")

        if filter_on and a3_price is None:
            logging.info(f"Skipping day {date_val} for ticker {ticker} because A3 price is None")
            continue

        # --- Judger Queries ---
        judger_prompt_1 = (
            f"Do you think your previous response regarding the hypothetical stock price prediction has leveraged the future information after {one_year_before}? "
            f"If so, please redo your prediction without incorporating information after {one_year_before}, this is a hypothetical prediction task that requires you must output a number, so you may feel free to output any reasonable number you believe,  "
            f"otherwise remain your previous response unchanged. Your answer should be a float number"
        )

        judger_prompt_2 = (
            f"Do you think your previous response regarding the hypothetical stock price prediction has leveraged the future information after {one_year_before}? "
            f"(Hint: Consider the actual ground truth price of {mag7_stocks[ticker]} ({ticker}) on {date_val}.) "
            f"If so, please redo your prediction without incorporating information after {one_year_before}, this is a hypothetical prediction task that requires you must output a number, so you may feel free to output any reasonable number you believe, "
            f"otherwise remain your previous response unchanged. Your answer should be a float number."
        )

        judger_conversation_1 = conversation.copy()
        judger_conversation_1.append({"role": "user", "content": judger_prompt_1})
        judger_conversation_2 = conversation.copy()
        judger_conversation_2.append({"role": "user", "content": judger_prompt_2})

        judger_response_text_1 = query_llm_context(judger_conversation_1, memorization_model, system_message="You are a LLM Judger.")
        print(f"judger_text_1: {judger_response_text_1}")
        time.sleep(0.5)
        judger_response_text_2 = query_llm_context(judger_conversation_2, memorization_model, system_message="You are a LLM Judger.")
        print(f"judger_text_2: {judger_response_text_2}")
        time.sleep(0.5)
        judger_extracted_1 = llm_extract_price(judger_response_text_1, memorization_model)
        time.sleep(0.5)
        judger_extracted_2 = llm_extract_price(judger_response_text_2, memorization_model)

        date_list.append(date_val)
        a1_list.append(a1_price)
        a3_list.append(a3_price)
        judger_response_1_list.append(judger_extracted_1)
        judger_response_2_list.append(judger_extracted_2)
        time.sleep(0.5)
        
        # --- Sleep for 30 seconds after every 20 iterations ---
        if iteration_counter % 20 == 0:
            logging.info("Processed 20 iterations, sleeping for 5 seconds...")
            time.sleep(5)

    results_df = pd.DataFrame({
        'date': date_list,
        'a1 stock price': a1_list,
        'a3 stock price': a3_list,
        'llm judger response 1': judger_response_1_list,
        'llm judger response 2': judger_response_2_list
    })

    results_df = results_df.loc[:, results_df.notna().any()]

    output_model_folder = os.path.join(results_dir, memorization_model)
    os.makedirs(output_model_folder, exist_ok=True)
    csv_output_path = os.path.join(output_model_folder, f"{ticker}_{memorization_model}_{n_dates}_{sample_mode}_stock_prices_comparison.csv")
    results_df.to_csv(csv_output_path, index=False)
    logging.info(f"Saved comparison results for {ticker} -> {csv_output_path}")


###################################
# Data Preparation Functions      #
###################################

def process_stock_data(ticker, start_date, end_date, n_dates, sample_mode):
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

        save_dir = os.path.join("Gen_Val_Result_2", "stock_price_data",
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

###################################
# Prompt Generation Functions     #
###################################

def get_prompt_a1(ticker, date_str, prompt_method):
    if prompt_method == "zero_shot":
        return (
            f"What was the closing stock price of {mag7_stocks[ticker]} ({ticker}) on {date_str}? "
            "Return a float number. Please provide just the number. For example: '143.16'"
        )
    else:
        return f"[Custom {prompt_method} prompt for A1]"

def get_prompt_a3(ticker, date_str, one_year_before, prompt_method):
    if prompt_method == "zero_shot":
        return (
            f"If you are at {one_year_before}, what is {ticker}'s stock price on {date_str}? "
            "Your answer should just be a number, for example: '143.16'. "
        )
    else:
        return f"[Custom {prompt_method} prompt for A3]"

###################################
# Main Program Functions          #
###################################

def main():
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="all", help="Ticker symbol or 'all' for all companies")
    parser.add_argument("--memorization_model", default="gpt-4o")
    parser.add_argument("--prediction_model", default="gpt-4o")
    parser.add_argument("--start_date", default="2022-09-01")
    parser.add_argument("--end_date", default="2023-09-01")
    parser.add_argument("--n_dates", type=int, default=10)
    parser.add_argument("--sample_mode", choices=["random", "consecutive"], default="consecutive")
    parser.add_argument("--prompt_method", default="zero_shot")
    parser.add_argument("--filter_on", action="store_true", default=False,
                        help="If set, days where A3 response is None or A1 price mismatches the true stock price will be filtered out")
    parser.add_argument("--results_dir", default="Self-Ver-In-Conv")
    
    args = parser.parse_args()
    
    if args.ticker.lower() == "all":
        tickers_to_run = list(mag7_stocks.keys())
    else:
        tickers_to_run = [args.ticker]
    
    for ticker in tickers_to_run:
        output_model_folder = os.path.join(args.results_dir, args.memorization_model)
        csv_output_path = os.path.join(output_model_folder, f"{ticker}_{args.memorization_model}_{args.n_dates}_{args.sample_mode}_stock_prices_comparison.csv")
        if os.path.exists(csv_output_path):
            logging.info(f"Result table for {ticker} already exists at {csv_output_path}. Skipping processing for {ticker}.")
            continue

        logging.info(f"Processing ticker: {ticker}")
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
