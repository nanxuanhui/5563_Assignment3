NewsCatalog = ['体育','娱乐','家居','彩票','房产','教育','时尚','时政','星座','游戏','社会','科技','股票','财经']

path = '/Users/nanxuan/Desktop/5563/Assignment3/combined_data/'
THUCNews_path = '/Users/nanxuan/Desktop/5563/Assignment3/data/THUCNews/'
SougouNews_path = '/Users/nanxuan/Desktop/5563/Assignment3/data/Sogounews/'

Data = open(path + 'Data.txt', 'a', encoding='utf-8')

for item in NewsCatalog:        # Merge THU dataset
    file = open(THUCNews_path + '{}.txt'.format(item), 'r', encoding='utf-8')
    txt = file.read().strip('\n').strip(' ')
    Data.write(txt + '\n')
    file.close()

print('THU data set merge completed')

for i in range(38):      # Merge Sogou News Data Set
    file = open(SougouNews_path + 'Sogounews_{}.txt'.format(i), 'r', encoding='utf-8')
    txt = file.read().strip('\n').strip(' ')
    Data.write(txt + '\n')
    file.close()

print('Sogou news data set merge completed')
