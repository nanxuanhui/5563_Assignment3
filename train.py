from gensim.models import Word2Vec
from gensim.models import word2vec
import logging

path = '/Users/nanxuan/Desktop/5563/Assignment3/combined_data/Data.txt'

sentences = word2vec.LineSentence(path)

#Training log
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# sg——Selection of two word2vec models. If it is 0, it is the CBOW model, if it is 1, it is the Skip-Gram model, and the default is 0, which is the CBOW model.
# hs——The choice between the two solution methods of word2vec. If it is 0, it is Negative Sampling. If it is 1 and the number of negative sampling is greater than 0, it is Hierarchical Softmax. The default is 0, which is Negative Sampling
# negative——The number of negative samples when using Negative Sampling, the default is 5. It is recommended to be between [3,10]
# min_count——The minimum word frequency required to calculate the word vector. This value can remove some very rare low-frequency words. The default is 5. If it is a small corpus, you can lower this value.
# epochs——The maximum number of iterations in the stochastic gradient descent method, the default is 5. For large corpus, this value can be increased
# alpha – initial step size for iterations in stochastic gradient descent. It is marked as η in the algorithm principle, and the default is 0.025.
# min_alpha - Since the algorithm supports gradually reducing the step size during the iteration process, min_alpha gives the minimum iteration step value.
model = Word2Vec(sentences, vector_size=40, window=5, epochs=500)
model.save('word2vec8.model')
