import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re  
import os
import numpy as np

import pdfplumber
from typing import Optional, List, Tuple
import pandas as pd
import psycopg2
import pickle
from sklearn.svm import SVC

import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from dotenv import load_dotenv

load_dotenv()

conn_details = {
        'dbname' : os.getenv('MY_DB_NAME'),
        'user': os.getenv('MY_DB_USER'),
        'password': os.getenv('MY_DB_PASSWORD'),
        'host': os.getenv('MY_DB_HOST'),
        'port': os.getenv('MY_DB_PORT')
}


def extract_date_from_text(text: str) -> Optional[datetime]:
    """
    Extracts datetime obj from text. 

    Args:
        text (str): The text we are extracting a date from. Expecting format of June 13, 2024 or June 13,2024 will be found somewhere in text.

    Returns:
        datetime object: datetime(year, month, day)
        None: None
    """

    pattern = r'(\w+ \d{1,2}, ?\d{4})'  
    match = re.search(pattern, text)
  
    if not match:
        return None
      
    date_str = match.group(1)  
    date_str = re.sub(r',(\d{4})', r', \1', date_str) 

    try:
        return datetime.strptime(date_str, '%B %d, %Y')
    except ValueError as ve:
        print(f"Unable to generate a datetime object: {ve}")
        return None





def get_latest_pdfs(url: str, last_known_date: datetime, end_date: Optional[datetime] = None) -> List[Tuple[datetime, str, str]]:
    """
    Scrapes the url and returns the PDF Links greater than last_known_date.
    Optionally, we can bound between last_known_date < PDF link <= end_date

    Args:
        url (str): "https://www.police.ucsd.edu/docs/reports/CallsandArrests/Calls_and_Arrests.asp"
        last_known_date (datetime object): datetime(year, month, day) 
        end_date (datetime object): datetime(year, month, day) 
    
    Returns:
        List[Tuple[datetime, str, str]]: List that contains 3-tuples. (datetime object, CallsForService/June 24, 2024.pdf, June 24, 2024.pdf)
        List[]: List can be empty
    """

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    non_empty_options = soup.select('option[value]:not([value=""])')

    new_pdfs = []

    for option in non_empty_options:
        text = option.text.strip()
        link_suffix = option['value']

        pdf_date = extract_date_from_text(text)

        if pdf_date is None:
            print(f"Could not extract a date from: {text}")
            continue

        if end_date:
            if last_known_date < pdf_date <= end_date:
                new_pdfs.append((pdf_date, link_suffix, text)) 
        else:
            if pdf_date > last_known_date:
                new_pdfs.append((pdf_date, link_suffix, text))

    new_pdfs.sort(key=lambda x: x[0]) 

    return new_pdfs


#returns the dates and the directory path of the pdf file
def download_pdfs(pdf_list: List[Tuple[datetime, str, str]], url_prefix: str, base_dir:str = 'pdfs') -> List[Tuple[datetime, str]]:
    """
    Returns a list of every pdf file downloaded. 

    Args:
        pdf_list (List[Tuple[datetime, str, str]]): (datetime object, CallsForService/June 24, 2024.pdf, June 24, 2024.pdf)
    Returns:
        List[Tuple[datetime, str]]: List that contains 2-tuples. (datetime object, path/to/pdf_file.pdf)
        List[]: List can be empty
    """

    total_pdf_dates_downloaded = []
    
    for pdf_info in pdf_list:
        pdf_date, pdf_link_suffix, pdf_name = pdf_info

        download_dir = os.path.join(base_dir, pdf_date.strftime('%Y'), pdf_date.strftime('%b'))
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        pdf_file_path = os.path.join(download_dir,  pdf_date.strftime('%m-%d-%Y.pdf'))


        total_pdf_dates_downloaded.append((pdf_date, pdf_file_path))


        full_download_url = os.path.join(url_prefix, pdf_link_suffix)
        response = requests.get(full_download_url)

        if response.status_code == 200:
            with open(pdf_file_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded PDF '{pdf_name}' as '{pdf_date.strftime('%m-%d-%Y.pdf')}' to '{download_dir}'")
        else:
            print(f"Failed to download '{pdf_name}'. Status code: {response.status_code}")

    return total_pdf_dates_downloaded





def list_files_in_directory(dir_path: str) -> List[Tuple[datetime, str]]:
    """
    Returns a list of every file in a directory of pdfs/ or cases/

    Args:
        dir_path (str): Path to directory
    Returns:
        List[Tuple[datetime, str]]: List that contains 3-tuples. (datetime object: (year, month, date),  path/to/pdf_file.pdf or path/to/pdf_file.txt)
        List[]: List can be empty
    """
    try:
        files_and_directories = os.listdir(dir_path)
    
        if "pdf" in dir_path:
            files = [ (datetime.strptime(f,"%m-%d-%Y.pdf"), os.path.join(dir_path, f)) for f in files_and_directories if os.path.isfile(os.path.join(dir_path, f))]
        else:
            files = [ (datetime.strptime(f,"%m-%d-%Y.txt"), os.path.join(dir_path, f)) for f in files_and_directories if os.path.isfile(os.path.join(dir_path, f))]
            
        files.sort(key=lambda x: x[0])
        return files
    except Exception as e:
        print(f"An error occurred in the list_files_in_directory function: {e}")
        return []
    




def generate_text_files(pdf_list: List[Tuple[datetime, str]], destination_dir = 'cases') -> List[Tuple[datetime, str]]:
    """
    Generates text files from input. 

    Args:
        pdf_list (List[Tuple[datetime, str]]): (datetime object: (year, month, day), path/to/pdf_file.pdf)
    Returns:
        List[Tuple[datetime, str]]: List that contains 2-tuples. (datetime object: (year, month, day), path/to/pdf.txt)
        List[]: List can be empty
    """
    txt_file_directories = []
    counter2 = -1
    for pdf_info in pdf_list:

        pdf_date, pdf_download_path = pdf_info

        destination = os.path.join(destination_dir,  pdf_date.strftime('%Y'), pdf_date.strftime('%b'))
        if not os.path.exists(destination):
            os.makedirs(destination)
        output_txt_file = os.path.join(destination, pdf_date.strftime('%m-%d-%Y.txt')) 

        txt_file_directories.append((pdf_date, output_txt_file))
    
        try: 
            with open(output_txt_file, 'w', encoding='utf-8') as file:
                with pdfplumber.open(pdf_download_path) as pdf:
                    for page in pdf.pages:
                        txt = page.extract_text()
                        txt = txt.replace('\xa0', ' ').replace('\u2010', '-')                        
                        txt = txt.split('\n')

                        for line in txt:
                            if "UCSD POLICE DEPARTMENT" in line or "CRIME AND FIRE LOG/MEDIA BULLETIN" in line:
                                counter2 = 0
                                continue
                            if counter2 != -1:
                                counter2 = -1
                                continue
                            if line == "\n":
                                continue
                            if line == " ":
                                continue

                            file.write(line)
                            file.write("\n")
                            
                            if 'Disposition' in line:
                                file.write(">")
                                file.write("\n")
        except IOError as e:
            print(f"An error occurred: {e}")

    return txt_file_directories



# CODE FROM PLAYGROUND.IPYNB 

def get_case_array(file_path):

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    cases = content.split('>')

    cases = [case.strip() for case in cases if case.strip()]
    
    return cases


def write_to_logger(case, date):
    logger_file_path = './weird_cases.txt'
    
    with open(logger_file_path, 'a', encoding='utf-8') as file:
        file.write(case + '\n')
        file.write(date.strftime('%m-%d-%Y') + '\n')
        file.write("------\n")
      

def filter_cases_function2(date):
    def filter_cases_fcn(case):

        lines = case.split('\n')
        
        if len(lines) < 8:
            print(f"Wrote to weird_cases.txt for: {date.strftime('%m-%d-%Y.txt')}")
            write_to_logger(case, date)
            return False
        elif lines[6].split(':')[-1] == '':
            return False
        else:
            return True
        
    return filter_cases_fcn

def filter_cases(filepath, date):
    return list(filter(filter_cases_function2(date), get_case_array(filepath)))



def normalize_case(case, date):
    # case = case.replace('\xa0', ' ').replace('\u2010', '-')
    case_lines = case.split('\n')
    case_dictionary = {}

    time_pattern = r'\b\d{1,2}:\d{2}\s[APM]{2}\b'
    time_match = re.findall(time_pattern, case_lines[5])
    if len(time_match) == 1:
        extracted_time = time_match[0]
    elif len(time_match) == 2:
        extracted_time = time_match[0] + ' - ' + time_match[-1]
    else:
        extracted_time = 'Unknown'


    if len(case_lines) == 8:
        case_dictionary['Incident Type'] = case_lines[0].strip()
        case_dictionary['Location'] = case_lines[1].strip()
        case_dictionary['Date'] = date
        case_dictionary['Time'] = extracted_time

        if "Summary:" in case_lines[6]:
            case_dictionary['Summary'] = case_lines[6].split('Summary:')[-1].strip()
        elif "Summary" in case_lines[6]:
            case_dictionary['Summary'] = case_lines[6].split('Summary')[-1].strip()
        elif "summary:" in case_lines[6]:
            case_dictionary['Summary'] = case_lines[6].split('summary:')[-1].strip()
        elif "summary" in case_lines[6]:
            case_dictionary['Summary'] = case_lines[6].split('summary')[-1].strip()
        else:
            case_dictionary['Summary'] = 'Someting wrong'
            print("Something wrong with case in normalize_case function")


        case_dictionary['Disposition'] = case_lines[7].split(':')[-1].strip()
    
    elif len(case_lines) > 8:
        case_dictionary['Incident Type'] = case_lines[0].strip()
        case_dictionary['Location'] = case_lines[1].strip()
        case_dictionary['Date'] = date
        case_dictionary['Time'] = extracted_time



        if "Summary:" in case_lines[6]:
            summary_str =  case_lines[6].split('Summary:')[-1]
        elif "Summary" in case_lines[6]:
            summary_str = case_lines[6].split('Summary')[-1]
        elif "summary:" in case_lines[6]:
            summary_str = case_lines[6].split('summary:')[-1]
        elif "summary" in case_lines[6]:
            summary_str = case_lines[6].split('summary')[-1]
        else:
            print("Something wrong with case in normalize_case function len > 8")
            print(case)
            summary_str = ""

        # summary_string = case_lines[6].split(':')[-1]
        i = 7
        while ("Disposition" not in case_lines[i]) and ("disposition" not in case_lines[i]):
            # print("this i passed: ", i)
            # print(case_lines[i])
            summary_str += " " + case_lines[i]
            i += 1

        case_dictionary['Summary'] = summary_str.strip()

        case_dictionary['Disposition'] = case_lines[i].split(':')[-1].strip()

    return case_dictionary


def turn_cases_into_objects(cases, date):

    dict_array = []
    for case in cases:
        case_dict = normalize_case(case, date)
        dict_array.append(case_dict)

    return dict_array


def print_case_objects(cases):
    case_number = 0
 
    for case_object in cases:
        print(f"Case #{case_number + 1}")
        for key, value in case_object.items():
            print(f"{key}: {value}")

        case_number += 1
        print('---')



def get_last_known_date(file_path: str) -> Optional[datetime]:
    """
    Pass in the txt file with last known date and will return a datetime object
    """
    try:
        with open(file_path, 'r') as file:
            last_known_date_str = file.read().strip()
            last_known_date = datetime.strptime(last_known_date_str, '%Y-%m-%d')
    except FileNotFoundError:
        last_known_date = datetime.now().date()

    return last_known_date



def get_all_cases(text_file_directories):

    all_cases = []

    for elem in text_file_directories:
        file_date, my_file_path = elem
        cases_with_summary = filter_cases(my_file_path, file_date)
        case_objects_array = turn_cases_into_objects(cases_with_summary, file_date.strftime('%-m/%-d/%Y'))
        all_cases.extend(case_objects_array)

    return all_cases



###### STEMMING/PORTING 

stop_words = set(stopwords.words('english'))

def remove_stop_words(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.lower() not in stop_words]  # Remove stop words
    return ' '.join(tokens)


stemmer = PorterStemmer()

def stem_text(text):
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


def data_cleaning_fcn(any_df):
   
    # Drop rows with missing values
    any_df = any_df.dropna().copy()

    # Apply lowercase transformation to 'Summary' and create 'tokenized' column
    any_df.loc[:, 'tokenized'] = any_df['Summary'].apply(lambda x: x.lower() if isinstance(x, str) else x)

    # Remove punctuation from 'tokenized'
    any_df.loc[:, 'tokenized'] = any_df['tokenized'].str.replace(f'[{string.punctuation}]', '', regex=True)

     # Remove stop words from 'tokenized'
    any_df.loc[:, 'tokenized'] = any_df['tokenized'].apply(remove_stop_words)

    # Apply stemming to 'tokenized'
    any_df.loc[:, 'tokenized'] = any_df['tokenized'].apply(stem_text)

    return any_df



def predict_labels(clean_crime_df):


    with open('./pickle_models/first_svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)

    with open('./pickle_models/tfidf_vectorizer.pickle', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

    tfidf_vectorized = tfidf_vectorizer.transform(clean_crime_df['tokenized'])
    predicted_y = svm_model.predict(tfidf_vectorized).astype(bool)

    return predicted_y


def prepare_for_psql(data_frame, predicted_y_values):

    
    data_frame = data_frame.drop(columns=['tokenized'])
    data_frame['Date'] = pd.to_datetime(data_frame['Date'], format='%m/%d/%Y').dt.date
    data_frame['Month_Year'] = data_frame['Date'].apply(lambda x: x.strftime('%b %Y'))
    data_frame['Interesting'] = predicted_y_values


    return data_frame



def write_to_psql(data_frame):

    try:
        
        conn = psycopg2.connect(**conn_details)
        cursor = conn.cursor()
    
        for _, row in data_frame.iterrows():
            cursor.execute("""
                INSERT INTO case_logs (
                    incident_type, location, date, time, summary, disposition, interesting, month_year
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
            """, (
                row['Incident Type'],
                row['Location'],
                row['Date'],
                row['Time'],
                row['Summary'],
                row['Disposition'],
                row['Interesting'],
                row['Month_Year']
            ))

        conn.commit()
        cursor.close()
        conn.close()

    except Exception as error:
        print(f"Error connecting to the database: {error}")


def csv_df_cleaning(df):
    df = df.copy()
    df['Interesting'] = df['Interesting'].apply(lambda x: 0 if np.isnan(x) == True else x)
    df['Bizarre'] = df['Bizarre'].apply(lambda x: 0 if np.isnan(x) == True else x)
    df['Interesting'] = df['Interesting'].apply(lambda x: bool(x))
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y').dt.date
    df['Month_Year'] = df['Date'].apply(lambda x: x.strftime('%b %Y'))
    df = df.dropna()
    return df

