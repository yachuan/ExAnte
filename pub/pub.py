import re
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from datetime import datetime
from collections import defaultdict
import xml.etree.ElementTree as ET
import os
import google.generativeai as genai
from datetime import datetime
from bs4 import BeautifulSoup
import openai
from datetime import datetime
from collections import defaultdict
import xml.etree.ElementTree as ET
import ast
import arxiv
import json
import pickle
from collections import Counter
from datetime import datetime
import pandas as pd
from collections import Counter
from googleapiclient.discovery import build



def name_to_id(paper_title):
    # Replace with your actual paper title
    # paper_title = "attention is all you need"

    # Define the API endpoint and parameters
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": paper_title,
        "fields": "title,paperId",
        "limit": 1  # Adjust as needed
    }

    # Make the API request
    response = requests.get(url, params=params)
    data = response.json()

    # Extract and print the paper ID
    if data.get("data"):
        paper = data["data"][0]
        # print(f"Title: {paper['title']}")
        # print(f"Semantic Scholar Paper ID: {paper['paperId']}")
        return paper['paperId']
    else:
        print("Paper not found in semantic.")
    

def get_citations_per_year(paper_id, limit=1000):
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
    params = {
        'fields': 'citingPaper.year',
        'limit': limit,  # Max number of citations to fetch
        # 'year': "2016-2018"
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Failed to get citations: {response.status_code}, {response.text}")

    data = response.json()
    citations = data.get('data', [])
    # print(citations)

    # Count citations by year
    years = [citation['citingPaper']['year'] for citation in citations if citation['citingPaper'].get('year')]
    citation_counts = Counter(years)

    # Sort by year
    sorted_citation_counts = dict(sorted(citation_counts.items()))
    return sorted_citation_counts

def complete_messages(role, prompt):
  return [
      {"role": "system", "content": "You are a helpful" + role},
      {"role": "user", "content": prompt}
  ]

openai.api_key = 'YOUR_API'

def generate_wc6(entry, cutoff_date1, year2):
    # prompt = f"Using the knowledge up to {cutoff_date}, Please summarize {entry} in a paragraph of about 100 words. (IMPORTANT: Any knowledge and information after {cutoff_date} is banned.)"
    # return prompt

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"You are a helpful assistant."},
            # {"role": "user", "content": f"Assume you are on {cutoff_date1}. Please list the most notable publications in the field of {entry} in computer science in {year2} that you consider impressive. Just list the achievements without explanation."+"For each publication, please provide title. Your output is ONLY a list as format: [{'title': 'article 1 name'}, {'title': 'article 2 name'}, ...]. You should avoid exporting publications that do not exist. Let's think it step by step."}
            # {"role": "user", "content": f"Assume you are on {cutoff_date1}. Please list the most notable publications in the field of {entry} in computer science in {year2} that you consider impressive. Just list the achievements without explanation."+"For each publication, please provide title. Your output is ONLY a list as format: [{'title': 'article 1 name'}, {'title': 'article 2 name'}, ...]. Since you should not know about any publications after the cutoff date, you should not provide any publications later than the cutoff date. You should avoid exporting publications that do not exist."}
            {"role": "user", "content": f"Assume you are on {cutoff_date1}. Please list the most notable publications in the field of {entry} in computer science published in {year2} that you consider impressive. Just list the achievements without explanation."+"For each publication, please provide title. Your output is ONLY a list as format: [{'title': 'article 1 name'}, {'title': 'article 2 name'}, ...]. You should avoid exporting publications that do not exist."}

        ],


        max_tokens = 1000, # Adjust based on how long you want the summary to be
        temperature = 1
    )
    response1 = response.choices[0].message.content

    response2 = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"You are a helpful assistant."},
            # {"role": "user", "content": f"Assume you are on {cutoff_date1}. Please list the most notable publications in the field of {entry} in computer science in {year2} that you consider impressive. Just list the achievements without explanation."+"For each publication, please provide title. Your output is ONLY a list as format: [{'title': 'article 1 name'}, {'title': 'article 2 name'}, ...]. You should avoid exporting publications that do not exist. Let's think it step by step."}
            # {"role": "user", "content": f"Assume you are on {cutoff_date1}. Please list the most notable publications in the field of {entry} in computer science in {year2} that you consider impressive. Just list the achievements without explanation."+"For each publication, please provide title. Your output is ONLY a list as format: [{'title': 'article 1 name'}, {'title': 'article 2 name'}, ...]. Since you should not know about any publications after the cutoff date, you should not provide any publications later than the cutoff date. You should avoid exporting publications that do not exist."}
            {"role": "user", "content": f"Assume you are on {cutoff_date1}. Please list the most notable publications in the field of {entry} in computer science published in {year2} that you consider impressive. Just list the achievements without explanation."+"For each publication, please provide title. Your output is ONLY a list as format: [{'title': 'article 1 name'}, {'title': 'article 2 name'}, ...]. You should avoid exporting publications that do not exist."},
            {"role": "assistant", "content": response1},
            # {"role": "user", "content": f"Do you think this answer relied on information beyond {cutoff_date1}? If you believe your response used future knowledge, please revise your answer based only on information available on or before {cutoff_date1}. Otherwise, you ONLY return 'No leakage'."}
            {"role": "user", "content": f"Do you think this answer relied on information beyond {cutoff_date1}? If you believe your response used future knowledge, please revise your answer based only on information available on or before {cutoff_date1}. Otherwise, you ONLY return 'No leakage'."}

        ],

        max_tokens = 1000, # Adjust based on how long you want the summary to be
        temperature = 1
    )
    response3 = response2.choices[0].message.content

    return (response1, response3)


def extract_between_brackets(input_string):
    start = input_string.find('[')
    end = input_string.find(']')

    if start != -1 and end != -1 and start < end:
        return input_string[start:end+1]
    else:
        return "No content found between brackets"
    

def merge_lists_no_duplicates(list1, list2):
    seen_titles = set()
    result = []

    for item in list1 + list2:
        original_title = item['title']
        normalized_title = re.sub(r'[^\w\s]', '', original_title.lower())

        if normalized_title not in seen_titles:
            seen_titles.add(normalized_title)
            result.append(item)

    return result

def date_str_to_int(list1):

    for item in list1:
      if item['year'] != '' and item['year'] != 'N/A':
        the_year = int(re.search(r'\d+', item['year']).group())
        item['year'] = the_year
    return list1




def get_arxiv_info(arxiv_id):
    client = arxiv.Client()

    search = arxiv.Search(id_list=[arxiv_id])

    # paper = next(search.results())
    paper = next(client.results(search))

    title = paper.title
    year = paper.published


    return title, year

def search_arxiv_by_title(title, max_results=5):

    client = arxiv.Client()

    # Create a more flexible search query
    search = arxiv.Search(
        query=f'(ti:"{title}" OR abs:"{title}")',  # Search in title and abstract
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    results = list(client.results(search))

    return results


def print_article_info(article):
    print(f"Title: {article.title}")
    print(f"Authors: {', '.join(author.name for author in article.authors)}")
    print(f"Published: {article.published}")
    print(f"URL: {article.entry_id}")
    print("\n")




def remove_by_arxiv(list_date2, the_cutoff):
    no_found_list = []
    arXiv_list = []
    for item in list_date2:
        tit = re.sub(r'[^\w\s]', '', item['title'].lower())
        if 'arxiv.org' in item['URL'].lower():
            arXiv_list.append(item)
            print(item)
            if 'astro-ph' in item['URL'].lower():
                IDs = re.search(r'arxiv\.org/(?:abs|pdf)/(?:[\w.-]+/)?(\d+)', item['URL']).group(1)

            else:
                IDs = re.search(r'(?:abs|pdf)/(\d+\.\d+)', item['URL']).group(1)


            title, year = get_arxiv_info(IDs)
            # print(f"Found {len(results)} results for '{search_title}':\n")
            # for article in results:

            #     print_article_info(article)
            if len(title) != 0:

                # print(re.sub(r'[^\w\s]', '', title.lower()) == tit)

                if re.sub(r'[^\w\s]', '', title.lower()) == tit:
                    if int(the_cutoff[:4]) < year:
                        print('Found the exist article but behind cutoff: ', item['title'], '\n')
                    # else:
                    #   print('satisfy article: ', item['title'], results[0].entry_id, '\n')
                    # print('yes')

                else:
                    # print('Not found article:', item, '\n')
                    no_found_list.append(item)
            else:
                # print('Not found article: ', item, '\n')
                # print_article_info(results[0])
                no_found_list.append(item)


        else:

            results = search_arxiv_by_title(tit)

            if len(results) != 0:
                if re.sub(r'[^\w\s]', '', results[0].title.lower()) == tit:
                    if int(the_cutoff[:4]) < results[0].published.year:
                        print('Found the exist article but behind cutoff: ', item['title'], '\n')
                        print('Found the exist article but behind cutoff: ', item, '\n')

                else:
                    # print('Not found article: ', item, '\n')
                    no_found_list.append(item)
            else:
                # print('Not found article: ', item, '\n')
                # print_article_info(results[0])
                no_found_list.append(item)

    return no_found_list, arXiv_list


def clean_json_string(json_string):
    # Remove any comments (both single-line and multi-line)
    json_string = re.sub(r'//.*?$|/\*.*?\*/', '', json_string, flags=re.MULTILINE | re.DOTALL)

    # Remove any trailing commas in objects and arra
    json_string = re.sub(r',\s*}', '}', json_string)
    json_string = re.sub(r',\s*]', ']', json_string)

    # Remove any non-printable characters
    json_string = ''.join(char for char in json_string if char.isprintable())

    # Remove any remaining non-JSON characters (keeping only valid JSON syntax)
    json_string = re.sub(r'[^\[\]{},:"0-9a-zA-Z\s.-]', '', json_string)

    # Parse and re-serialize to ensure valid JSON
    try:
        parsed_json = json.loads(json_string)
        return json.dumps(parsed_json, indent=2)
    except json.JSONDecodeError as e:
        return f"Error: Unable to parse JSON. {str(e)}"



# publication_search
def search_paper_year1(i):
    api_key = "YOUR_API"
    cse_id = "YOUR_ID"

    result_list = []


    # Create a service object for the Custom Search API
    service = build("customsearch", "v1", developerKey=api_key)

    # Execute the search
    result = service.cse().list(q=i, cx=cse_id).execute()
    result_list.append(i)
    for kk in result["items"]:
        result_list.append(kk['title'])
        result_list.append(kk['link'])
    # year_list = []
    # title_list = []

    # print(result_list)
    return result_list

# publication_search
def search_paper_year(query):
    api_key = "YOUR_API"
    cse_id = "YOUR_ID"

    result_list = []



    for i in query:

        # Create a service object for the Custom Search API
        service = build("customsearch", "v1", developerKey=api_key)

        # Execute the search
        result = service.cse().list(q=i['title'], cx=cse_id).execute()
        # print(result)
        if 'items' in result:
            result_list.append(i['title'])
            result_list.append(result["items"][0]['title'])
            result_list.append(result["items"][0]['link'])
    # year_list = []
    # title_list = []

    # print(result)
    return result_list

# ps2
def search_paper_year2(i):
    api_key = "YOUR_API"
    cse_id = "YOUR_ID"

    result_list = []


    # Create a service object for the Custom Search API
    service = build("customsearch", "v1", developerKey=api_key)

    # Execute the search
    result = service.cse().list(q=i, cx=cse_id).execute()
    # print(result)
    if "items" in result:
        result_list.append(i)
        for kk in result["items"][:1]:
            result_list.append(kk['title'])
            result_list.append(kk['link'])
    # year_list = []
    # title_list = []

    # print(result_list)
    return result_list

# arixivps3
def search_paper_year3(i):
    api_key = "YOUR_API"
    cse_id = "YOUR_ID"

    result_list = []


    # Create a service object for the Custom Search API
    service = build("customsearch", "v1", developerKey=api_key)

    # Execute the search
    result = service.cse().list(q=i, cx=cse_id).execute()
    if 'items' in result:
        result_list.append(i)
        for kk in result["items"][:1]:
            result_list.append(kk['title'])
            result_list.append(kk['link'])
    # year_list = []
    # title_list = []

    # print(result_list)
    return result_list



def convert_date_range(date_str):
    if date_str == None or date_str == '':
        return date_str
    months = {
        "January": "01", "February": "02", "March": "03", "April": "04",
        "May": "05", "June": "06", "July": "07", "August": "08",
        "September": "09", "October": "10", "November": "11", "December": "12"
    }
    if len(date_str) > 24:
        date_str = date_str.split("-")[0]

    # Single-date conversion
    if "-" not in date_str:
        day, month, year = date_str.split()
        # print(date_str)
        return f"{year}-{months[month]}-{int(day):02d}"

    # Range-date conversion
    else:
        start, end_month_year = date_str.split("-")
        start_day = int(start)
        end_day, end_month, end_year = end_month_year.split()
        start_month = end_month
        start_year = end_year

        if start_day > int(end_day):  # Adjust the start month if needed
            start_month_index = list(months.values()).index(months[end_month]) - 1
            start_month = list(months.keys())[start_month_index]

        start_date = f"{start_year}-{months[start_month]}-{start_day:02d}"
        end_date = f"{end_year}-{months[end_month]}-{int(end_day):02d}"
        return start_date



from urllib.request import Request, urlopen
def ieee(url):
    # URL of the IEEE Xplore document
    # url = "https://ieeexplore.ieee.org/document/7445236"

    # Set headers to mimic a browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36"
    }

    # Send a GET request to the URL
    response = requests.get(url, headers=headers)
    # print(response.status_code)
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the page content
        soup = BeautifulSoup(response.content, "html.parser")
        # print(soup)
        # print(soup.find_all("script"))
        # print(len(soup.find_all("script")))


        # for script in soup.find_all("script"):

        #     results = re.search("var xplGlobal = '(.*)'", script.text)
        #     if results:
        #         print('search:', results[1])

        # Look for the publication date element (depends on the page structure)
        # IEEE Xplore pages often include JSON-LD data, which contains publication details
        scripts = soup.find_all("script")
        # print(scripts)
        for script in scripts:
            if script.string != None:
                data = script.string
                # print(data)
                if "displayPublicationDate" in data:
                    # print('Tryeeeee')
                    start_index = data.find("displayPublicationDate") + len("displayPublicationDate") + 3
                    end_index = data.find('"', start_index)
                    publication_date = data[start_index:end_index]
                    # print("Publication Date:", publication_date)
                    return convert_date_range(publication_date)
                

def acmdl(url=''):
    # URL of the IEEE Xplore document
    # url = """https://dl.acm.org/doi/10.1145/3677178"""

    # Set headers to mimic a browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36"
    }

    # Send a GET request to the URL
    response = requests.get(url, headers=headers)
    # print(response.status_code)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the page content
        soup = BeautifulSoup(response.content, "html.parser")
        scripts = soup.find(id='core-history')
        data = str(scripts)
        # print(scripts)

        start_index = data.find("Published") + len("Published") + 6
        end_index = data.find('<', start_index)
        publication_date = data[start_index:end_index]
        # print("Publication Date:", publication_date)

        return convert_date_range(publication_date)
    


def iacr(url1):
    # URL of the IEEE Xplore document
    # url = """https://dl.acm.org/doi/10.1145/3677178"""
    url = url1
    # Set headers to mimic a browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36"
    }

    # Send a GET request to the URL
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the page content
        soup = BeautifulSoup(response.content, "html.parser")
        scripts = soup.find(id='metadata')
        data = str(scripts)
        # print(data)

        start_index = data.find("received") - 12
        end_index = data.find('received', start_index) - 2
        publication_date = data[start_index:end_index]
        # print("Publication Date:", publication_date)

        return publication_date
    

def acl(url1):
    # URL of the IEEE Xplore document
    # url = """https://dl.acm.org/doi/10.1145/3677178"""
    url = url1
    # Set headers to mimic a browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36"
    }

    # Send a GET request to the URL
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the page content
        soup = BeautifulSoup(response.content, "html.parser")
        scripts = soup.find(id='main')
        data = str(scripts)
        # print(data)

        start_index = data.find("Month:") + 15
        end_index = data.find('</dd>', start_index)
        publication_date = data[start_index:end_index]

        start_index1 = data.find("Year:") + 14
        end_index1 = data.find('</dd>', start_index1)
        publication_date1 = data[start_index1:end_index1]
        # print("Publication Date:", publication_date)
        if '–' in publication_date:
            publication_date = publication_date.split("–")[0]
        return convert_date_range("1"+" "+publication_date+" "+publication_date1)
    

def mlr(url1):
    # URL of the IEEE Xplore document
    # url = """https://dl.acm.org/doi/10.1145/3677178"""
    url = url1
    # Set headers to mimic a browser request
    if url.count('/') > 3:
        url = re.sub(r'/[^/]+$', '', url)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36"
    }

    # Send a GET request to the URL
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the page content
        soup = BeautifulSoup(response.content, "html.parser")
        scripts = soup.find('h2')
        data = str(scripts)
        # print(url)


        data = data[data.index(':') + 1:].strip()

        # Split by comma and look for the part containing the date
        parts = data.split(',')

        # The date is typically in the second part (index 1)
        # We'll strip any leading/trailing whitespace
        date_part = parts[1].strip()
        return convert_date_range(date_part)
    


from datetime import date
def search_paper_year_LD_to_Date(search_paper_year_LD):
    for i in range(0, len(search_paper_year_LD)):
        search_paper_year_LD[i] = search_paper_year_LD[i].replace(" ...", "")
        if 'https:/' in search_paper_year_LD[i] or 'http:/' in search_paper_year_LD[i]:
            continue
        else:
            search_paper_year_LD[i] = re.sub(r'[^\w\s]', ' ', search_paper_year_LD[i]).lower()
            search_paper_year_LD[i] = ' '.join(word for word in search_paper_year_LD[i].split())
    publication_date_list = []
    for i in range(0, len(search_paper_year_LD)):
        if 'https:/' in search_paper_year_LD[i] or 'http:/' in search_paper_year_LD[i]:
            if (search_paper_year_LD[i-1] in search_paper_year_LD[i-2]) or (search_paper_year_LD[i-2] in search_paper_year_LD[i-1]):
                publication_date = ''
                if 'arxiv' in search_paper_year_LD[i] and (search_paper_year_LD[i][23]).isdigit():
                    IDs = re.search(r'(?:abs|pdf)/(\d+\.\d+)', search_paper_year_LD[i]).group(1)
                    publication_date_o = get_arxiv_info(IDs)[1]
                    publication_date0 = (publication_date_o.year, publication_date_o.month, publication_date_o.day)
                    publication_date = date(*publication_date0).strftime("%Y-%m-%d")
                    # print(publication_date.year)
                    # print(publication_date.month)
                    # print(publication_date.day)
                    # print(1)
                elif 'ieeexplore.ieee.' in search_paper_year_LD[i]:
                    publication_date = ieee(search_paper_year_LD[i])
                    # print(2)
                elif 'dl.acm.org' in search_paper_year_LD[i]:
                    publication_date = acmdl(search_paper_year_LD[i])
                    # print(3)
                elif 'eprint.iacr.org' in search_paper_year_LD[i]:
                    search_paper_year_LD[i] = search_paper_year_LD[i].replace('.pdf', '')
                    publication_date = iacr(search_paper_year_LD[i])
                    # print(4)
                elif 'acl' in search_paper_year_LD[i]:
                    search_paper_year_LD[i] = search_paper_year_LD[i].replace('.pdf', '')
                    publication_date = acl(search_paper_year_LD[i])
                    # print(5)
                # elif 'sciencedirect' in search_paper_year_LD[i]:
                #     publication_date = sciencedirect(search_paper_year_LD[i])
                #     # print(5)
                # elif 'springer' in search_paper_year_LD[i]:
                #     publication_date = springer(search_paper_year_LD[i])
                #     # print(5)
                # elif 'amazon' in search_paper_year_LD[i]:
                #     publication_date = amazon(search_paper_year_LD[i])
                # elif 'thecvf' in search_paper_year_LD[i]:
                #     arxiv_url = search_paper_year3(search_paper_year_LD[i-1])[2]
                #     # print('arxiv_ur: ', arxiv_url)
                #     IDs = re.search(r'(?:abs|pdf|html)/(\d+\.\d+)', arxiv_url).group(1)
                #     publication_date_o = get_arxiv_info(IDs)[1]
                #     publication_date = (publication_date_o.year, publication_date_o.month, publication_date_o.day)
                #     # print(6)

                elif "proceedings.mlr.press" in search_paper_year_LD[i]:
                    if "pdf" in search_paper_year_LD[i]:
                        mlrID = "/" + search_paper_year_LD[i].split('/')[-1].replace('.pdf', '')
                        search_paper_year_LD[i] = search_paper_year_LD[i].replace('.pdf', '.html')
                        count = search_paper_year_LD[i].count(mlrID)
                        if count > 1:
                            search_paper_year_LD[i] = search_paper_year_LD[i].replace(mlrID, "", 1)
                        mlrID_html = mlrID + '.html'
                        search_paper_year_LD[i] = search_paper_year_LD[i].replace(mlrID_html, "")
                    else:
                        mlrID = "/" + search_paper_year_LD[i].split('/')[-1].replace('.html', '')
                        count = search_paper_year_LD[i].count(mlrID)
                        if count > 1:
                            search_paper_year_LD[i] = search_paper_year_LD[i].replace(mlrID, "", 1)
                        mlrID_html = mlrID + '.html'
                        search_paper_year_LD[i] = search_paper_year_LD[i].replace(mlrID_html, "")
                    publication_date = mlr(search_paper_year_LD[i])

                elif "usenix" in search_paper_year_LD[i]:
                    th1 = search_paper_year2(search_paper_year_LD[i-1])
                    if th1 != []:
                        acm_url = th1[2]
                        publication_date = acmdl(acm_url)

                    else:
                        acm_url = []
                        publication_date = ''

                elif "neurips" in search_paper_year_LD[i]:
                    th1 = search_paper_year2(search_paper_year_LD[i-1])
                    if th1 != []:
                        acm_url = th1[2]
                        publication_date = acmdl(acm_url)

                    else:
                        acm_url = []
                        publication_date = ''

                elif "nips" in search_paper_year_LD[i]:
                    th1 = search_paper_year2(search_paper_year_LD[i-1])
                    if th1 != []:
                        acm_url = th1[2]
                        publication_date = acmdl(acm_url)

                    else:
                        acm_url = []
                        publication_date = ''

                else:
                    publication_date = ''
                    # print(i)
                    # print(search_paper_year_LD[i])
                    # print(publication_date_list)

                    # th1 = search_paper_year2(search_paper_year_LD[i-1])

                    # if th1 != []:
                    #     acm_url = th1[2]
                    #     publication_date = acmdl(acm_url)

                    # else:
                    #     acm_url = []
                    #     publication_date = []

                    # print(7)
                publication_date_list.append(search_paper_year_LD[i-2])
                publication_date_list.append(search_paper_year_LD[i-1])
                publication_date_list.append(search_paper_year_LD[i])
                publication_date_list.append(publication_date)

    return publication_date_list




def name_to_date(paper_title):
    # Replace with your actual paper title
    # paper_title = "attention is all you need"

    # Define the API endpoint and parameters
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": paper_title,
        "fields": "title,publicationDate",
        "limit": 1  # Adjust as needed
    }

    # Make the API request
    response = requests.get(url, params=params)
    data = response.json()
    # print(data)

    # Extract and print the paper ID

    return data['data'][0]['publicationDate']


def is_older(date1, date2):
    # Parse the strings into datetime objects
    d1 = datetime.strptime(date1, "%Y-%m-%d")
    d2 = datetime.strptime(date2, "%Y-%m-%d")

    # Compare the dates
    return d1 < d2


def count_leakage(leakage_num, total_num, leakage_rate, add_leakage_rate, LD2, year2):
    total_num0 = 0
    leakage_num0 = 0
    leakage_rate0 = 0
    LD3 = []
    if len(LD2) > 0:

        for index, i in enumerate(LD2):
            if i != None:
                if bool(re.match(r"^\d{4}-\d{2}-\d{2}$", i)):
                    if is_older(year2+'-'+'07'+'-'+'01', i):
                        total_num0 += 1
                        leakage_num0 += 1
                    else:
                        total_num0 += 1
                        LD3.append(LD2[index-2])
        if total_num0 > 0:

            leakage_rate0 = leakage_num0/total_num0

            total_num += total_num0
            leakage_num += leakage_num0
            leakage_rate = leakage_num/total_num
            add_leakage_rate += leakage_rate0


            print('total num: ', total_num0)
            print('leakage num: ', leakage_num0)
            print('leakage_rate: ', leakage_rate0)

            print("")

            print('Accumulative leakage num: ', leakage_num)
            print("Accumulative total num: ", total_num)
            print('Accumulative leakage rate: ', leakage_rate)
            print('Add leakage rate: ', add_leakage_rate)

            return leakage_num, total_num, leakage_rate, add_leakage_rate, LD3

        else:
            return leakage_num, total_num, leakage_rate, add_leakage_rate, LD3
    else:
            return leakage_num, total_num, leakage_rate, add_leakage_rate, LD3
    



def main():

    openai.api_key = 'YOUR_API'
    leakage_num = 0
    total_num = 0
    leakage_rate = 0
    add_leakage_rate = 0

    LD1_list = []
    LD2_list = []
    LD3_list = []


    # Read CSV file into DataFrame
    df = pd.read_csv('exante_pub.csv')

    # Convert DataFrame to list of rows
    data = df.values.tolist()

    for field_year in data:
        list_date14 = []
        LD = []
        # field = 'GPU'
        field = field_year[0]

        # year1 = 'July 1, 2019'
        # year2 = '2019'
        year1 = 'July 1, ' + field_year[1]
        year2 = field_year[1]

        for k in range(0, 3):
            try:
                events_org = generate_wc6(field, year1, year2)

                if 'No leakage' in events_org[1]:
                    events = events_org[0]
                else:
                    events = events_org[1]
                # events = generate_wc6(field, year1, year2)
                # print(events)
                # list1 = find_title(events)
                list1 = extract_between_brackets(events)

                list1.replace("null", "''")
                list1 = re.sub(r'// .*', '', list1)


                list1_1 = ast.literal_eval(list1)
                for i in range(0, len(list1_1)):
                    if {} in list1_1:
                        list1_1.remove({})

                LD = merge_lists_no_duplicates(LD, list1_1)

            except Exception as e:
                print("An error occurred")


            # LD = date_str_to_int(LD)

        # list_date14 = LD.copy()

        # print(len(LD))
        LD1 = search_paper_year(LD)
        # LD1
        LD1_list.append(LD1)

        LD2 = search_paper_year_LD_to_Date(LD1)

        for index, title1_title2_date_url in enumerate(LD2):
            if title1_title2_date_url == '':
                pub_date = name_to_date(LD2[index-2])
                LD2[index] = pub_date

        # print(len(LD2)/4)
        # print(year1)
        # LD2
        LD2_list.append(LD2)
        # no_leakage_paper = len(LD2)/4

        leakage_num, total_num, leakage_rate, add_leakage_rate, LD3 = count_leakage(leakage_num, total_num, leakage_rate, add_leakage_rate, LD2, year2)

        good_pub = len(LD3)
        for i in range(0, len(LD3)):
            # Example usage
            paper_id = name_to_id(LD3[i])  # Replace with actual paper ID
            citations_per_year = get_citations_per_year(paper_id)
            for pubYear, count in citations_per_year.items():
                # print(pubYear, count)
                if pubYear != year2 and count < 1:
                    good_pub -= 1
                    break
        LD3_result = f"field name: {field}, date:{year1}, total paper: {len(LD1)/3}, good paper: {good_pub} percent: {good_pub/(len(LD1)/3)}"
        print(LD3_result)
        LD3_list.append(LD3_result)

    # save file
    with open('Paper_title.pkl', 'wb') as f:
        pickle.dump(LD2_list, f)
    
    with open('Pub_result.pkl', 'wb') as f1:
        pickle.dump(LD3_list, f1)





if __name__ == "__main__":
    main()