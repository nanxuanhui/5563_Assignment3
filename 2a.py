import numpy as np
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

# Set up Matplotlib to support Chinese characters before doing any plotting
plt.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus'] = False  # Used to display negative signs normally

# Load model
glove_input_file = '/Users/nanxuan/Desktop/5563/Assignment3/chinese_wiki_embeding20000.txt'
word2vec_model = KeyedVectors.load_word2vec_format('/Users/nanxuan/Desktop/5563/Assignment3/baike_26g_news_13g_novel_229g.bin', binary=True)
glove_model = KeyedVectors.load_word2vec_format(glove_input_file, binary=False, no_header=True)
model = Word2Vec.load("/Users/nanxuan/Desktop/5563/Assignment3/word2vec3.model")
my_word2vec_model = model.wv

# choose word
words = ['清华大学', '爱因斯坦', '加拿大']

# Get embed
word2vec_embeddings = np.array([word2vec_model[word] for word in words if word in word2vec_model])
glove_embeddings = np.array([glove_model[word] for word in words if word in glove_model])
my_word2vec_embeddings = np.array([my_word2vec_model[word] for word in words if word in my_word2vec_model])

# Define a function to perform t-SNE and plot
def tsne_transform(embeddings, title, ax):
    if len(embeddings) > 1:
        tsne = TSNE(n_components=2, perplexity=min(len(embeddings)-1, 30))
        embeddings_2d = tsne.fit_transform(embeddings)
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
        for i, word in enumerate(words):
            ax.annotate(word, xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]))
        ax.set_title(title)
    else:
        print(f"Not enough words for {title} model.")

# Visualization
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
tsne_transform(word2vec_embeddings, 'Word2Vec', axs[0])
tsne_transform(glove_embeddings, 'GloVe', axs[1])
tsne_transform(my_word2vec_embeddings, 'My Word2Vec', axs[2])

plt.show()

