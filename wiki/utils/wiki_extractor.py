import requests
import pandas as pd
import os
import time
from typing import Tuple
import re
from bs4 import BeautifulSoup
import csv
import openai

def get_revision_id(title, target_date):
    url = "https://en.wikipedia.org/w/api.php"
    
    # Convert target_date to int and subtract 1 to get the end of previous year
    previous_year = int(target_date) - 1
    
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "revisions",
        "rvlimit": 1,  # We only need the closest revision
        "rvprop": "ids|timestamp",
        "rvdir": "older",  # Get the first revision before the date
        "rvstart": f"{previous_year}1231235959"  # End of the previous year YYYYMMDDHHMMSS
    }
    
    try:
        print(f"Finding revision for {title} before end of {previous_year} (cutoff year: {target_date})")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'query' not in data:
            print(f"No query data in response for {title}")
            return None
            
        pages = data["query"]["pages"]
        if not pages:
            print(f"No pages found for {title}")
            return None
            
        page_id = list(pages.keys())[0]
        if page_id == '-1':
            print(f"Page not found for {title}")
            return None
        
        if "revisions" in pages[page_id]:
            revision = pages[page_id]["revisions"][0]  # Get the first (closest) revision
            print(f"Found revision from {revision.get('timestamp', 'unknown date')}")
            return revision["revid"]
        
        print(f"No revisions found for {title}")
        return None
            
    except Exception as e:
        print(f"Error getting revision for {title}: {str(e)}")
        return None

def get_wiki_content_by_revision(title, revision_id=None):
    url = "https://en.wikipedia.org/w/api.php"
    
    if revision_id:
        # If we have a revision ID, use it without the page parameter
        params = {
            "action": "parse",
            "format": "json",
            "oldid": revision_id,
            "prop": "text|sections",
            "formatversion": "2"
        }
    else:
        # For current content, use the page parameter
        params = {
            "action": "parse",
            "format": "json",
            "page": title,
            "prop": "text|sections",
            "formatversion": "2"
        }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'error' in data:
            print(f"API Error: {data['error']}")
            return None
            
        if 'parse' not in data:
            print(f"No parse data found for {title}")
            return None
            
        # Get HTML content
        html_content = data['parse']['text']
        
        # Convert HTML to plain text
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove unwanted elements
        unwanted_elements = [
            'table', 'div.infobox', 'div.navbox', 'div.sidebar',
            'div.metadata', 'div.hatnote', 'div.thumb', 'div.toc',
            'script', 'style', 'sup.reference', 'span.mw-editsection'
        ]
        for element in soup.select(', '.join(unwanted_elements)):
            element.decompose()
        
        # Only keep paragraph content
        paragraphs = []
        for p in soup.find_all('p'):
            # Skip empty paragraphs or those with only whitespace/references
            text = p.get_text(strip=True)
            if text and not text.startswith('[') and len(text) > 50:  # Minimum length to avoid fragments
                paragraphs.append(text)
        
        if not paragraphs:
            print(f"No valid paragraph content found for {title}")
            return None
        
        # Join paragraphs with newlines
        text_content = '\n\n'.join(paragraphs)
        
        # Clean up the text
        text_content = re.sub(r'\[[\d\s,]+\]', '', text_content)  # Remove reference numbers
        text_content = re.sub(r'\s+', ' ', text_content)  # Normalize whitespace
        
        return text_content if text_content.strip() else None
        
    except Exception as e:
        print(f"Error getting content for {title}: {str(e)}")
        return None

def save_content(content, folder_path, title, year):
    if not content:
        print(f"No content to save for {title} ({year})")
        return
        
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    filename = f"{title}_{year}_content.txt"
    file_path = os.path.join(folder_path, filename)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Saved {len(content)} characters to {file_path}")
    except Exception as e:
        print(f"Error saving content for {title}: {str(e)}")

def is_temporal_title_gpt(title: str) -> Tuple[bool, str]:
    """
    Use GPT to determine if a topic has significant post-2010 evolution and identify a cutoff year.
    Returns: (is_temporal, explanation with cutoff year if valid)
    """
    prompt = f"""You are an expert in analyzing the historical evolution of topics. Analyze if the topic "{title}" meets these specific criteria:

1. Post-2010 Evolution: Must have significant development/changes after 2010 (new research, technological advances, shifts in discourse)
2. Identifiable Cutoff: Must have a clear time point where the topic saw a noticeable shift in development/methodology/adoption
3. Distinct Phases: Must have considerable discussion both before and after the cutoff, with clear differences in understanding

Respond in this exact format:
Classification: VALID or INVALID
Cutoff Year: [YYYY] (only if VALID)
Reason: Brief explanation including key developments that justify the classification and cutoff year

Example response for "Deep Learning":
Classification: VALID
Cutoff Year: 2012
Reason: AlexNet in 2012 marked revolutionary shift in deep learning. Pre-2012: limited adoption, basic neural networks. Post-2012: explosion in research, GPU acceleration, widespread industry adoption.

Example response for "Ancient Rome":
Classification: INVALID
Reason: Historical topic with established facts, no significant post-2010 developments or methodological shifts."""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        result = response.choices[0].message['content']
        classification = "VALID" in result.split("Classification:")[1].split("\n")[0]
        
        # Extract cutoff year and reason
        cutoff_year = None
        if classification and "Cutoff Year:" in result:
            cutoff_year = result.split("Cutoff Year:")[1].split("\n")[0].strip()
            cutoff_year = re.search(r'\d{4}', cutoff_year).group(0)
            
        reason = result.split("Reason:")[1].strip()
        
        # Include cutoff year in reason if valid
        if classification and cutoff_year:
            return True, f"Cutoff {cutoff_year}: {reason}"
        return False, reason
        
    except Exception as e:
        print(f"Error classifying {title}: {str(e)}")
        return False, str(e) 