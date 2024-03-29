import re
import jieba

stopwords_table_path = '/Users/nanxuan/Desktop/5563/Assignment3/四川大学机器智能实验室停用词库.txt'
file = open(stopwords_table_path, 'r', encoding='utf-8')
stopwords_table = file.readlines()
stopwords_list = []
for item in stopwords_table:
    stopwords_list.append(item.replace('\n',''))        # Create a stop word list

NewsCatalog = ['体育','娱乐','家居','彩票','房产','教育','时尚','时政','星座','游戏','社会','科技','股票','财经']

file_path = '/Users/nanxuan/Desktop/5563/Assignment3/data/THUCNews/'

i = 0
for category in NewsCatalog:
    combine = open(file_path + '{}.txt'.format(category), 'w', encoding='utf-8')
    sentence = []
    while(True):
        if(i % 200 == 0):
            print('\rNumber of texts processed：{}'.format(i), end='')
        try:
            file = open(file_path + category + '/' + '{}.txt'.format(i), 'r', encoding='utf-8')
            i += 1
            txt = file.read().replace('\n　　',' ')      # 一篇文章为一排
            file.close()
            txt = ''.join(re.findall('[\u4e00-\u9fa5| |]', txt))
            txt = ' '.join(jieba.cut(txt, cut_all=False)).replace('   ',' ')
            for word in txt.split(' '):
                for stopword in stopwords_list:
                    if word == stopword:
                        txt = txt.replace(stopword+' ','')
            sentence.append(txt+'\n')
        except:
            combine.write(''.join(sentence))
            print(category + 'Text processing completed')
            break
