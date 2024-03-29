#COMP-SCI 5563 Assignment 3
#From dat file to txt file

dat_file_path = '/Users/nanxuan/Desktop/5563/Assignment3/data/news_tensite_xml.dat'
txt_file_path = '/Users/nanxuan/Desktop/5563/Assignment3/data/news_tensite_xml.txt'

with open(dat_file_path, 'rb') as file:  # Read in binary mode
    binary_content = file.read()
    text_content = binary_content.decode('ISO-8859-1')  # Decode text using ISO-8859-1 encoding

with open(txt_file_path, 'w', encoding='utf-8') as file:
    file.write(text_content)

