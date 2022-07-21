# -*- coding: utf-8 -*-

# Yu Qin
# 04282022

### This file contains utility function that will be used in to extract data from 10-K
### Part of the code is from the following link: 
### https://gist.github.com/anshoomehra/ead8925ea291e233a5aa2dcaa2dc61b2

from bs4 import BeautifulSoup
import re
import pandas as pd

def Parse_10k(file_path):

    with open(file_path) as fp:
        raw_10k = fp.read()
    
    # Regex to find <DOCUMENT> tags
    doc_start_pattern = re.compile(r'<DOCUMENT>')
    doc_end_pattern = re.compile(r'</DOCUMENT>')
    # Regex to find <TYPE> tag prceeding any characters, terminating at new line
    type_pattern = re.compile(r'<TYPE>[^\n]+')

    # Create 3 lists with the span idices for each regex

    ### There are many <Document> Tags in this text file, each as specific exhibit like 10-K, EX-10.17 etc
    ### First filter will give us document tag start <end> and document tag end's <start> 
    ### We will use this to later grab content in between these tags
    doc_start_is = [x.end() for x in doc_start_pattern.finditer(raw_10k)]
    doc_end_is = [x.start() for x in doc_end_pattern.finditer(raw_10k)]

    ### Type filter is interesting, it looks for <TYPE> with Not flag as new line, ie terminare there, with + sign
    ### to look for any char afterwards until new line \n. This will give us <TYPE> followed Section Name like '10-K'
    ### Once we have have this, it returns String Array, below line will with find content after <TYPE> ie, '10-K' 
    ### as section names
    doc_types = [x[len('<TYPE>'):] for x in type_pattern.findall(raw_10k)]

    document = {}

    # Create a loop to go through each section type and save only the 10-K section in the dictionary
    for doc_type, doc_start, doc_end in zip(doc_types, doc_start_is, doc_end_is):
        if doc_type == '10-K':
            document[doc_type] = raw_10k[doc_start:doc_end]
    
    # Write the regex to find Item tags
    regex = re.compile(r'(>Item(\s|&#160;|&nbsp;)(1A|1B|7A|7|8)\.{0,1})|(ITEM\s(1A|1B|7A|7|8))')

    # Use finditer to math the regex
    matches = regex.finditer(document['10-K'])

    # Matches
    matches = regex.finditer(document['10-K'])

    # Create the dataframe
    test_df = pd.DataFrame([(x.group(), x.start(), x.end()) for x in matches])

    test_df.columns = ['item', 'start', 'end']
    test_df['item'] = test_df.item.str.lower()

    # Get rid of unnesesary charcters from the dataframe
    test_df.replace('&#160;',' ',regex=True,inplace=True)
    test_df.replace('&nbsp;',' ',regex=True,inplace=True)
    test_df.replace(' ','',regex=True,inplace=True)
    test_df.replace('\.','',regex=True,inplace=True)
    test_df.replace('>','',regex=True,inplace=True)

    # Drop duplicates
    pos_dat = test_df.sort_values('start', ascending=True).drop_duplicates(subset=['item'], keep='last')

    # Set item as the dataframe index
    pos_dat.set_index('item', inplace=True)

    # Get Item 1a
    item_1a_raw = document['10-K'][pos_dat['start'].loc['item1a']:pos_dat['start'].loc['item1b']]
    # Get Item 7
    item_7_raw = document['10-K'][pos_dat['start'].loc['item7']:pos_dat['start'].loc['item7a']]
    # Get Item 7a
    item_7a_raw = document['10-K'][pos_dat['start'].loc['item7a']:pos_dat['start'].loc['item8']]

    # Convert the raw text we have to exrtacted to BeautifulSoup object 
    item_1a_content = BeautifulSoup(item_1a_raw, 'lxml')
    item_7_content = BeautifulSoup(item_7_raw, 'lxml')
    item_7a_content = BeautifulSoup(item_7a_raw, 'lxml')

    # Delete the tables from content
    for i in item_1a_content.find_all('table'):
        i.decompose()
    for i in item_7_content.find_all('table'):
        i.decompose()
    for i in item_7a_content.find_all('table'):
        i.decompose()
    
    # Generate paragraphs list from the content
    item_1a_list = [text for text in item_1a_content.stripped_strings]
    item_7_list = [text for text in item_7_content.stripped_strings]
    item_7a_list = [text for text in item_7a_content.stripped_strings]

    # Merge short paragraphs into one and divide into sentences
    item_1a_list = "".join(item_1a_list).split('. ')
    item_7_list = "".join(item_7_list).split('. ')
    item_7a_list = "".join(item_7a_list).split('. ')

    # Generate dictionary and return
    dict_ = {'item1a': item_1a_list, 'item7': item_7_list, 'item7a': item_7a_list}
    return dict_