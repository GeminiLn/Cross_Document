# -*- coding: utf-8 -*-

# Yu Qin
# 04282022

### This file contains utility function that will be used in to extract data from 10-Q
### Part of the code is from the following link: 
### https://gist.github.com/anshoomehra/ead8925ea291e233a5aa2dcaa2dc61b2

from bs4 import BeautifulSoup
import re
import pandas as pd

def Parse_10q(file_path):

    with open(file_path) as fp:
        raw_10q = fp.read()
    
    # Regex to find <DOCUMENT> tags
    doc_start_pattern = re.compile(r'<DOCUMENT>')
    doc_end_pattern = re.compile(r'</DOCUMENT>')
    # Regex to find <TYPE> tag prceeding any characters, terminating at new line
    type_pattern = re.compile(r'<TYPE>[^\n]+')

    # Create 3 lists with the span idices for each regex

    ### There are many <Document> Tags in this text file, each as specific exhibit like 10-K, EX-10.17 etc
    ### First filter will give us document tag start <end> and document tag end's <start> 
    ### We will use this to later grab content in between these tags
    doc_start_is = [x.end() for x in doc_start_pattern.finditer(raw_10q)]
    doc_end_is = [x.start() for x in doc_end_pattern.finditer(raw_10q)]

    ### Type filter is interesting, it looks for <TYPE> with Not flag as new line, ie terminare there, with + sign
    ### to look for any char afterwards until new line \n. This will give us <TYPE> followed Section Name like '10-K'
    ### Once we have have this, it returns String Array, below line will with find content after <TYPE> ie, '10-K' 
    ### as section names
    doc_types = [x[len('<TYPE>'):] for x in type_pattern.findall(raw_10q)]

    document = {}

    # Create a loop to go through each section type and save only the 10-K section in the dictionary
    for doc_type, doc_start, doc_end in zip(doc_types, doc_start_is, doc_end_is):
        if doc_type == '10-Q':
            document[doc_type] = raw_10q[doc_start:doc_end]

    # Write the regex to find Item tags
    regex = re.compile(r'(>Item(\s|&#160;|&nbsp;)(1A|1|2|3|4)\.{0,1})|(ITEM\s(1A|1|2|3|4))|(PART\s(II|I))')

    # Use finditer to math the regex
    matches = regex.finditer(document['10-Q'])

    # Matches
    matches = regex.finditer(document['10-Q'])

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

    # Drop redundent item labels from the document index
    for i in range(test_df.shape[0]):
        if test_df['item'][i] == 'parti':
            break
    test_df = test_df.drop(index=[j for j in range(i)]).reset_index(drop=True)

    # Rename the items by part(1/2)
    part = 0
    for i in range(test_df.shape[0]):

        if test_df['item'][i] == 'parti':
            part = 1
        if test_df['item'][i] == 'partii':
            part = 2
        
        test_df.iloc[i, 0] = test_df['item'][i] + '_' + str(part)

    # Drop part item
    test_df.drop(index=test_df[test_df['item']=='parti_1'].index[0], inplace=True)
    test_df.drop(index=test_df[test_df['item']=='partii_2'].index[0], inplace=True)
    pos_dat = test_df.reset_index(drop=True)
    # If the document does not contain item 2 in section 2, we need to ignore the item2_2 in the following code
    if 'item2_2' not in pos_dat['item'].values:
        no_item2_2 = 1
    else:
        no_item2_2 = 0

    # Set item as the dataframe index
    pos_dat.set_index('item', inplace=True)

    # Get Item 1_1
    item_1_1_raw = document['10-Q'][pos_dat['start'].loc['item1_1']:pos_dat['start'].loc['item2_1']]
    # Get Item 1_2
    item_1_2_raw = document['10-Q'][pos_dat['start'].loc['item2_1']:pos_dat['start'].loc['item3_1']]
    # Get Item 1_3
    item_1_3_raw = document['10-Q'][pos_dat['start'].loc['item3_1']:pos_dat['start'].loc['item4_1']]
    # Get Item 2_1A
    if no_item2_2:
        pass
    else:
        item_2_1A_raw = document['10-Q'][pos_dat['start'].loc['item1a_2']:pos_dat['start'].loc['item2_2']]

    # Convert the raw text we have to exrtacted to BeautifulSoup object 
    #item_1_1_content = BeautifulSoup(item_1_1_raw, 'lxml')
    item_1_2_content = BeautifulSoup(item_1_2_raw, 'lxml')
    item_1_3_content = BeautifulSoup(item_1_3_raw, 'lxml')
    if no_item2_2:
        pass
    else:
        item_2_1A_content = BeautifulSoup(item_2_1A_raw, 'lxml')

    # Delete tables from the content
    for i in item_1_2_content.find_all('table'):
        i.decompose()
    for i in item_1_3_content.find_all('table'):
        i.decompose()
    if no_item2_2:
        pass
    else:
        for i in item_2_1A_content.find_all('table'):
            i.decompose()

    # Generate paragraphs list from the content
    item_1_2_list = [text for text in item_1_2_content.stripped_strings]
    item_1_3_list = [text for text in item_1_3_content.stripped_strings]
    if no_item2_2:
        pass
    else:
        item_2_1A_list = [text for text in item_2_1A_content.stripped_strings]

    # Merge short paragraphs into one and divide into sentences
    item_1_2_list = "".join(item_1_2_list).split('. ')
    item_1_3_list = "".join(item_1_3_list).split('. ')
    if no_item2_2:
        pass
    else:
        item_2_1A_list = "".join(item_2_1A_list).split('. ')

    # Generate dictionary and return
    if no_item2_2:
        dict_ = {'item1_2': item_1_2_list, 'item1_3': item_1_3_list}
    else:
        dict_ = {'item1_2': item_1_2_list, 'item1_3': item_1_3_list, 'item2_1A': item_2_1A_list}
    return dict_