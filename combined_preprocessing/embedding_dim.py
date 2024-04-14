from gensim.models import KeyedVectors

file_path = '/Users/nanxuan/Desktop/5563/Assignment3/baike_26g_news_13g_novel_229g.bin'

try:
    model = KeyedVectors.load_word2vec_format(file_path, binary=True)
except UnicodeDecodeError:
    model = KeyedVectors.load_word2vec_format(file_path, binary=False)

embedding_dim = model.vector_size
print(f'The embedding dimensions are: {embedding_dim}')
