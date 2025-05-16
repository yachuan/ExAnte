import google.generativeai as genai
import re
import requests
from datetime import datetime
from bs4 import BeautifulSoup
import openai
from datetime import datetime
from collections import defaultdict
import ast
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import re
from datetime import datetime, timedelta
import re
import pickle


def add_month(date_str):
    """
    Subtracts one month from a given date string in 'YYYY-MM-DD' format.

    Args:
        date_str (str): Date string in 'YYYY-MM-DD' format

    Returns:
        str: Date string one month prior in 'YYYY-MM-DD' format

    Raises:
        ValueError: If date string format is invalid
    """
    # Validate date format
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        raise ValueError("Date must be in 'YYYY-MM-DD' format")

    try:
        # Convert string to datetime
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')

        # Subtract one month
        new_date = date_obj + relativedelta(months=1)

        # Convert back to string
        return new_date.strftime('%Y-%m-%d')
    except ValueError as e:
        raise ValueError(f"Invalid date: {e}")

# Example usage:



def subtract_days(date_str, days=365):
    """
    Subtracts specified number of days from a date string in 'YYYY-MM-DD' format.

    Args:
        date_str (str): Date string in 'YYYY-MM-DD' format
        days (int): Number of days to subtract (default: 10)

    Returns:
        str: Date string with subtracted days in 'YYYY-MM-DD' format
    """
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        raise ValueError("Date must be in 'YYYY-MM-DD' format")

    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        new_date = date_obj - timedelta(days=days)
        return new_date.strftime('%Y-%m-%d')
    except ValueError as e:
        raise ValueError(f"Invalid date: {e}")
    


def generate_wc6(cutoff_date1, question):

    genai.configure(api_key='YOUR_API')

    generation_config = {
      "temperature": 1,
      "top_p": 0.95,
      "top_k": 40,
      "max_output_tokens": 8192,
      "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
      model_name="gemini-1.5-pro",
      generation_config=generation_config,
      system_instruction="You are a helpful AI assistant.",
    )

    chat_session = model.start_chat(
      history=[
        {
          "role": "user",
          "parts": [
            f"Assume you are on {cutoff_date1}. Now please answer the following question. Just leave the best answer without explanation. " + question + "You should answer this question honestly. If you don't know, please answer I don't know. Please don't guess or speculate.",

          ],
        },
      ]
    )

    response = chat_session.send_message("INSERT_INPUT_HERE")

    res1 = response.text



    chat_session2 = model.start_chat(
      history=[
        {
          "role": "user",
          "parts": [
            f"Assume you are on {cutoff_date1}. Now please answer the following question. Just leave the best answer without explanation. " + question + "You should answer this question honestly. If you don't know, please answer I don't know. Please don't guess or speculate.",

          ],
        },
        {
          "role": "model",
          "parts": [
            res1,
          ],
        },
        {
          "role": "user",
          "parts": [
            f"Do you think this answer relied on information beyond {cutoff_date1}? If you believe your response used future knowledge, please revise your answer based only on information available on or before {cutoff_date1}. Otherwise, you ONLY return 'No leakage'.",
          ],
        },
      ]
    )

    response2 = chat_session2.send_message("INSERT_INPUT_HERE")

    res2 = response2.text

    # print('response1: ', res1)

    # print('response3: ', res2)



    return res1, res2

def main():
    total_num_ = 0
    leakage_num_ = 0
    df = pd.csv('exante_qa.csv')
    data = df.values.tolist()

    res_list = []

    for year in data:
        print(len(year[1]))


        n = 0
        unknow_list = []
        for i in year[1]:
            n += 1
 

            ans0 = generate_wc6(add_month(i[0]), a)
            known = str(i[0]) in str(ans0)
            if known is False:
                unknow_list.append(n)
            # print(n, ans0, known)

        # print("test: ")

        n = 0
        leakage_num = 0
        for i in year[1]:
            n += 1
            if n in unknow_list:
                continue
            a = """\n**Question: On what date"""+i[1]+"""?:

            \nYou should provide answer follow the format of YYYY-MM-DD

            """
            ans00 = generate_wc6(subtract_days(i[0]), a)
            if 'No leakage' in ans00[1]:
                ans = ans00[0]
            else:
                ans = ans00[1]
            leakage = "I don't know" not in str(ans)
            if leakage:
                leakage_num += 1
            # print(n, ans, leakage)



        print(n-len(unknow_list))
        print(leakage_num)
        res_list.append(n-len(unknow_list), leakage_num)
        total_num_ += (n-len(unknow_list))
        leakage_num_ += leakage_num
        
    with open('result_QA.pkl', 'wb') as f:
        pickle.dump(res_list, f)
    

if __name__ == "__main__":
    main()