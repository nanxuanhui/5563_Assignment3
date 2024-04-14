import re
import jieba

print("The program starts running...")

# Load stop words
stopwords_table_path = '/Users/nanxuan/Desktop/5563/Assignment3/四川大学机器智能实验室停用词库.txt'
print(f"Start loading the stop word list: {stopwords_table_path}")
with open(stopwords_table_path, 'r', encoding='utf-8') as file:
    stopwords_list = [line.strip() for line in file]  # Create a stop word list
print("The stop words are loaded. The total number of stop words loaded is：", len(stopwords_list))

# Prepare to process text data
path = '/Users/nanxuan/Desktop/5563/Assignment3/data/'
news_file_path = path + 'news_tensite_xml.txt'
print(f"Start processing news text data: {news_file_path}")

with open(news_file_path, 'r', encoding='gb18030') as file:
    news = []
    i = 0
    for txt in file:
        txt = ''.join(re.findall('[\u4e00-\u9fa5|\n]', txt))
        txt = ' '.join(jieba.cut(txt, cut_all=False))

        # Stop word filtering
        words = txt.split(' ')
        txt = ' '.join(word for word in words if word not in stopwords_list)

        news.append(txt)
        if i > 0 and i % 30000 == 0:
            out_file_path = path + f'Sogounews_{int(i/30000-1)}.txt'
            with open(out_file_path, 'w', encoding='utf-8') as out_file:
                out_file.write(''.join(news))
            news = []  # Clear saved news content and prepare for the next batch


            print(f"Processed and saved {i} news items to file: {out_file_path}")
        i += 1
    if news:  # Make sure the last batch of data is also written to the file
        out_file_path = path + f'Sogounews_{int(i/30000)}.txt'
        with open(out_file_path, 'w', encoding='utf-8') as out_file:
            out_file.write(''.join(news))
        print(f"The last batch of news has been processed and saved to file：{out_file_path}")

print("All news are processed, the total number of news processed：", i)