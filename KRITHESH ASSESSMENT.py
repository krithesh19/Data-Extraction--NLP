#!/usr/bin/env python
# coding: utf-8

# In[8]:


pip install beautifulsoup4 requests pandas


# In[10]:


import pandas as pd
from bs4 import BeautifulSoup
import requests

# Load URLs from input.xlsx
df = pd.read_excel(r"C:\Users\krithesh\Desktop\Input.xlsx")
urls = df['URL'].tolist()

# Function to extract article text from URL
def extract_text_from_url(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad requests

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract article title
        title = soup.title.text.strip() if soup.title else 'No Title'

        # Extract article text
        article_text = ''
        article_body = soup.find('body')
        if article_body:
            paragraphs = article_body.find_all('p')
            article_text = ' '.join([p.text.strip() for p in paragraphs])

        return title, article_text

    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return None, None

# Iterate through URLs, extract content, and save to text files
for url in urls:
    url_id = url.split("/")[-1].split(".")[0]  # Extract URL_ID from the URL
    title, article_text = extract_text_from_url(url)

    if title and article_text:
        # Save content to a text file
        with open(f"{url_id}.txt", 'w', encoding='utf-8') as file:
            file.write(f"Title: {title}\n\n")
            file.write(f"{article_text}\n\n")
        print(f"Successfully extracted and saved content for {url}")
    else:
        print(f"Failed to extract content for {url}")

print("Extraction and text file creation complete.")


# In[13]:


pip install chardet


# In[17]:


import requests
import chardet
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
from textblob import TextBlob

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Function to extract article text from a URL
def extract_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Extract only the article text, excluding header, footer, etc.
    article_text = ' '.join([p.get_text() for p in soup.find_all('p')])
    return article_text

# Function to calculate the number of syllables in a word
def syllable_count(word):
    vowels = "aeiouy"
    count = 0
    word = word.lower()
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count

# Function to perform text analysis
def analyze_text(text):
    # Load stop words
    stopwords_files = [
        "C:\\Users\\krithesh\\Desktop\\StopWords_Currencies.txt",
        "C:\\Users\\krithesh\\Desktop\\StopWords_DatesandNumbers.txt",
        "C:\\Users\\krithesh\\Desktop\\StopWords_Generic.txt",
        "C:\\Users\\krithesh\\Desktop\\StopWords_GenericLong.txt",
        "C:\\Users\\krithesh\\Desktop\\StopWords_Geographic.txt",
        "C:\\Users\\krithesh\\Desktop\\StopWords_Names.txt",
        "C:\\Users\\krithesh\\Desktop\\StopWords_Auditor.txt",
    ]
    custom_stopwords = set()
    for file_path in stopwords_files:
        with open(file_path, 'rb') as file:
            result = chardet.detect(file.read())
            encoding = result['encoding']
        with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
            custom_stopwords.update(file.read().splitlines())

    # Load positive and negative words
    positive_words_file = "C:\\Users\\krithesh\\Desktop\\positive-words.txt"
    negative_words_file = "C:\\Users\\krithesh\\Desktop\\negative-words.txt"
    
    with open(positive_words_file, 'r', encoding='utf-8', errors='ignore') as file:
        positive_words = file.read().splitlines()
    
    with open(negative_words_file, 'r', encoding='utf-8', errors='ignore') as file:
        negative_words = file.read().splitlines()

    # Clean the text
    cleaned_text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)  # Remove punctuation

    # Tokenize the text
    words = nltk.word_tokenize(cleaned_text)

    # Remove stop words
    words = [word for word in words if word.lower() not in custom_stopwords]

    # Perform text analysis
    positive_score = sum(1 for word in words if word in positive_words)
    negative_score = sum(1 for word in words if word in negative_words)
    
    total_words = len(words)
    avg_sentence_length = total_words / len(nltk.sent_tokenize(text))
    percentage_complex_words = sum(1 for word in words if syllable_count(word) > 2) / total_words
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    avg_words_per_sentence = total_words / len(nltk.sent_tokenize(text))
    complex_word_count = sum(1 for word in words if syllable_count(word) > 2)
    syllables_per_word = sum(syllable_count(word) for word in words) / total_words
    personal_pronouns = sum(1 for word in words if word.lower() in ["i", "we", "my", "ours", "us"])
    avg_word_length = sum(len(word) for word in words) / total_words

    # Calculate sentiment scores using TextBlob
    blob = TextBlob(cleaned_text)
    polarity_score = blob.sentiment.polarity
    subjectivity_score = blob.sentiment.subjectivity

    # Create a dictionary with computed variables
    result_dict = {
        'POSITIVE SCORE': positive_score,
        'NEGATIVE SCORE': negative_score,
        'POLARITY SCORE': polarity_score,
        'SUBJECTIVITY SCORE': subjectivity_score,
        'AVG SENTENCE LENGTH': avg_sentence_length,
        'PERCENTAGE OF COMPLEX WORDS': percentage_complex_words,
        'FOG INDEX': fog_index,
        'AVG NUMBER OF WORDS PER SENTENCE': avg_words_per_sentence,
        'COMPLEX WORD COUNT': complex_word_count,
        'WORD COUNT': total_words,
        'SYLLABLE PER WORD': syllables_per_word,
        'PERSONAL PRONOUNS': personal_pronouns,
        'AVG WORD LENGTH': avg_word_length
    }

    return result_dict

# Load URLs from the input Excel file
input_file_path = "C:\\Users\\krithesh\\Desktop\\Input.xlsx"
df_input = pd.read_excel(input_file_path)
output_data = []

# Process each row in the input file
for index, row in df_input.iterrows():
    url_id = row['URL_ID']
    url = row['URL']

    # Extract text from the URL
    article_text = extract_text(url)

    # Perform text analysis
    result_dict = analyze_text(article_text)

    # Add additional information
    result_dict['URL_ID'] = url_id
    result_dict['URL'] = url

    # Append the result to the output data list
    output_data.append(result_dict)

# Create a DataFrame from the output data
df_output = pd.DataFrame(output_data)

# Save the output to the specified Excel file
output_file_path = "C:\\Users\\krithesh\\Desktop\\Output results.xlsx"
df_output.to_excel(output_file_path, index=False)


# # Here how I Approach to the Solution:
# 

# DATA EXTRACTION:
# 
# *BeautifulSoup and Requests: Used BeautifulSoup for HTML parsing and requests to fetch the webpage content.
# 
# *Extracting Article Text: The extract_text_from_url function extracts the article text by considering only the paragraphs 
#  within the body of the HTML
#  
# TEXT ANALYSIS:
# 
# Stop Words and Custom Dictionaries: Loaded custom stop words from various files specified in the input.
# Tokenization and Cleaning: Tokenized the text, removed stop words, and cleaned the text by removing extra whitespaces and punctuation.
#     
# SENTIMENTAL ANALYTICS: Used TextBlob for sentiment analysis, calculating polarity and subjectivity scores.
#     
# CALCULATING VARIOUS METRICS:
#     
# *Positive and Negative Scores
# 
# *Polarity and Subjectivity Scores
# 
# *Average Sentence Length
# 
# *Percentage of Complex Words
# 
# *Fog Index
# 
# *Average Number of Words per Sentence
# 
# *Complex Word Count
# 
# *Word Count
# 
# *Syllables per Word
# 
# *Personal Pronouns
# 
# *Average Word Lengt

# DEPENDENCIES:
# 
# Ensure you have the required libraries installed
# 

# EXECUTION:
# 
# *Save the provided code into a Python file (e.g., analyze_text_data.py)
# 
# *Run the Python file in your terminal or command prompt:

# In[ ]:




