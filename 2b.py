import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from gensim.models import Word2Vec, KeyedVectors
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load and preprocess data
dataset_folder_path = '/Users/nanxuan/Desktop/5563/Assignment3/data/THUCNews'
texts, labels = [], []
print("Starting data loading...")
for file_name in os.listdir(dataset_folder_path):
    if file_name.endswith('.txt'):
        label = file_name[:-4]
        file_path = os.path.join(dataset_folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            texts.append(file.read())
            labels.append(label)
        print(f"Loaded {label}")
print("Data loading complete. Starting preprocessing...")

# Tokenize the text and prepare training data
print("Tokenizing texts...")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=100)  # Assuming max length of text is 100

# Encode labels
print("Encoding labels...")
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
labels = np.asarray(encoded_labels)
unique_labels = np.unique(labels)
if len(unique_labels) > 2:  # Multi-class classification
    y = to_categorical(encoded_labels)
    output_units = len(unique_labels)
    activation_function = 'softmax'
    loss_function = 'categorical_crossentropy'

for label in unique_labels:
    print(f"Class {label}: {np.sum(encoded_labels == label)} samples")


# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)

# Function to load or create embeddings
def get_embedding_matrix(word_index, embeddings_index, embedding_dim=300, train_new=True):
    print("Creating embedding matrix...")
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        # Use the word as the key to get its embedding vector
        embedding_vector = embeddings_index[word] if word in embeddings_index else None
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


# Load pre-trained Word2Vec and GloVe embeddings
print("Loading pre-trained embeddings...")
pretrained_w2v_path = '/Users/nanxuan/Desktop/5563/Assignment3/baike_26g_news_13g_novel_229g.bin'
pretrained_glove_path = '/Users/nanxuan/Desktop/5563/Assignment3/chinese_wiki_embeding20000.txt'
pretrained_w2v = KeyedVectors.load_word2vec_format(pretrained_w2v_path, binary=True)
pretrained_glove = KeyedVectors.load_word2vec_format(pretrained_glove_path, binary=False, no_header=True)

# Get embedding matrices
print("Getting embedding matrices...")
w2v_embedding_matrix = get_embedding_matrix(word_index, pretrained_w2v, 128, False)
glove_embedding_matrix = get_embedding_matrix(word_index, pretrained_glove, 300, False)
#self_trained_w2v_matrix = get_embedding_matrix(word_index, embedding_dim=300, train_new=True)

# Function to build and train the model
def build_and_train_model(X_train, y_train, X_test, y_test, embedding_matrix):
    embedding_dim = embedding_matrix.shape[1]
    model = Sequential()
    model.add(Embedding(input_dim=embedding_matrix.shape[0],
                        output_dim=embedding_dim,
                        weights=[embedding_matrix],
                        input_length=100,
                        trainable=False))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.5))  # 添加dropout层
    model.add(LSTM(50))
    model.add(Dropout(0.5))  # 添加dropout层
    model.add(Dense(output_units, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128)
    return model


def print_model_performance(model, X_train, y_train, X_test, y_test):
    # Evaluate the model on training and test data
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Print performance metrics
    print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


# Train and evaluate models with different embeddings
#print("Training with self-trained Word2Vec embeddings...")
#model_self_w2v = build_and_train_model(X_train, y_train, X_test, y_test, self_trained_w2v_matrix)

# Train the Word2Vec model and print its performance
model_w2v = build_and_train_model(X_train, y_train, X_test, y_test, w2v_embedding_matrix)
print("Results for Word2Vec model:")
print_model_performance(model_w2v, X_train, y_train, X_test, y_test)

# Train the GloVe model and print its performance
model_glove = build_and_train_model(X_train, y_train, X_test, y_test, glove_embedding_matrix)
print("Results for GloVe model:")
print_model_performance(model_glove, X_train, y_train, X_test, y_test)