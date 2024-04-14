import numpy as np
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

# Set up Matplotlib to support Chinese characters before doing any plotting
plt.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus'] = False  # Used to display negative signs normally

# Load models
word2vec_model = KeyedVectors.load_word2vec_format('/Users/nanxuan/Desktop/5563/Assignment3/baike_26g_news_13g_novel_229g.bin', binary=True)
glove_model = KeyedVectors.load_word2vec_format('/Users/nanxuan/Desktop/5563/Assignment3/chinese_wiki_embeding20000.txt', binary=False, no_header=True)
model = Word2Vec.load("/Users/nanxuan/Desktop/5563/Assignment3/word2vec3.model")
my_word2vec_model = model.wv

# Original list of words
original_words = ['清华大学', '爱因斯坦', '加拿大', '北京大学', '复旦大学', '交通大学', '师范大学', '南京大学', '武汉大学',
                  '浙江大学', '中山大学', '南开大学', '天津大学', '厦门大学', '东南大学', '华中科技大学', '中南大学',
                  '哈尔滨工业大学', '北京航空航天大学', '北京理工大学', '北京科技大学', '北京邮电大学', '北京林业大学',
                  '北京工业大学', '北京交通大学', '北京化工大学', '北京工商大学', '北京体育大学', '北京医科大学', '北京师范大学',
                  '北京外国语大学', '北京语言大学', '北京邮电大学', '北京联合大学', '北京中医药大学', '北京建筑大学',
                  '北京电影学院', '北京舞蹈学院', '北京青年政治学院', '北京服装学院', '北京石油化工学院', '北京农学院',
                  '北京协和医学院', '北京体育学院', '北京大学', '清华大学', '北京交通大学', '北京工业大学', '北京航空航天大学',
                  '北京理工大学', '北京科技大学', '北京林业大学', '北京邮电大学', '北京师范大学', '北京外国语大学', '北京语言大学',
                  '北京邮电大学', '北京联合大学', '北京中医药大学', '北京建筑大学', '北京电影学院', '北京舞蹈学院',
                  '北京青年政治学院', '北京服装学院', '北京石油化工学院', '北京农学院', '北京协和医学院', '北京体育学院',
                    '北京交通大学', '北京工业大学', '北京航空航天大学', '北京理工大学']

# Function to filter embeddings and words
def filter_embeddings(words, model):
    filtered_words = [word for word in words if word in model]
    embeddings = np.array([model[word] for word in filtered_words])
    return embeddings, filtered_words

# Get embeddings and filter words
word2vec_embeddings, filtered_word2vec_words = filter_embeddings(original_words, word2vec_model)
glove_embeddings, filtered_glove_words = filter_embeddings(original_words, glove_model)
my_word2vec_embeddings, filtered_my_word2vec_words = filter_embeddings(original_words, my_word2vec_model)

# Define a function to perform t-SNE and plot
def tsne_transform(embeddings, words, title, ax):
    if len(embeddings) > 1:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(len(embeddings)-1, 30))
        embeddings_2d = tsne.fit_transform(embeddings)
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
        for i, word in enumerate(words):
            ax.annotate(word, xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]))
        ax.set_title(title)
    else:
        print(f"Not enough words for {title} model.")

# Visualization
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
tsne_transform(word2vec_embeddings, filtered_word2vec_words, 'Word2Vec', axs[0])
tsne_transform(glove_embeddings, filtered_glove_words, 'GloVe', axs[1])
tsne_transform(my_word2vec_embeddings, filtered_my_word2vec_words, 'My Word2Vec', axs[2])

plt.show()


