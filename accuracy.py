from gensim.models import Word2Vec

model = Word2Vec.load('word2vec8.model')


print(model.vector_size)
print(model.total_train_time)
print(model.wv)
print(model.wv.most_similar('清华大学'))
print(model.wv.most_similar('狗'))
print(model.wv.most_similar('爱因斯坦'))
print(model.wv.most_similar('加拿大'))