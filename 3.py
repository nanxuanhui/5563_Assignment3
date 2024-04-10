import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

file_path = '/Users/nanxuan/Desktop/5563/Assignment3/combined_data/Data.txt'

print("加载数据...")

# 读取文本文件，将连续行组成句子对
sentences = []
with open(file_path, 'r', encoding='utf-8') as file:
    prev_line = file.readline().strip()
    for line in file:
        line = line.strip()
        if line:  # 确保行不为空
            sentences.append((prev_line, line))
            prev_line = line

# 将句子对列表转换为DataFrame
data = pd.DataFrame(sentences, columns=['sentence1', 'sentence2'])

print("数据加载完成，前几行如下:")
print(data.head())

# 加载分词器和模型
print("加载BERT分词器和模型...")
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')
print("BERT分词器和模型加载完成。")

# 定义函数以使用GloVe生成句子嵌入
def glove_sentence_embedding(sentence, glove_model):
    print(f"计算GloVe嵌入: {sentence[:30]}...")
    if isinstance(sentence, str):
        words = sentence.split()  # Split the sentence into words
        word_embeddings = [glove_model[word] for word in words if word in glove_model]
        if word_embeddings:
            return np.mean(word_embeddings, axis=0)
    return np.zeros(glove_model.vector_size)

# 定义函数以使用BERT生成句子嵌入
def bert_sentence_embedding(sentence):
    print(f"计算BERT嵌入: {sentence[:30]}...")
    encoded_input = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**encoded_input)
    embeddings = output.last_hidden_state
    attention_mask = encoded_input['attention_mask']
    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    sentence_embedding = sum_embeddings / sum_mask
    return sentence_embedding[0].numpy()

# 加载GloVe模型
print("加载GloVe模型...")
glove_input_file = '/Users/nanxuan/Desktop/5563/Assignment3/chinese_wiki_embeding20000.txt'
glove_model = KeyedVectors.load_word2vec_format(glove_input_file, binary=False, no_header=True)
print("GloVe模型加载完成。")

# 对数据集中的每个句子对生成GloVe和BERT嵌入
print("生成GloVe嵌入...")
data['glove_embedding1'] = data['sentence1'].apply(lambda x: glove_sentence_embedding(x, glove_model))
data['glove_embedding2'] = data['sentence2'].apply(lambda x: glove_sentence_embedding(x, glove_model))

print("生成BERT嵌入...")
data['bert_embedding1'] = data['sentence1'].apply(bert_sentence_embedding)
data['bert_embedding2'] = data['sentence2'].apply(bert_sentence_embedding)

# 定义函数以计算余弦相似度
def calculate_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

# 计算相似度
print("计算相似度...")
data['glove_similarity'] = data.apply(lambda x: calculate_similarity(x['glove_embedding1'], x['glove_embedding2']), axis=1)
data['bert_similarity'] = data.apply(lambda x: calculate_similarity(x['bert_embedding1'], x['bert_embedding2']), axis=1)

# 查看相似度结果
print("相似度计算完成，结果如下:")
print(data[['glove_similarity', 'bert_similarity']].head())


