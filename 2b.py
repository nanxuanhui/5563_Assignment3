import os
import numpy as np
from keras.layers import Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from gensim.models import Word2Vec, KeyedVectors
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Load and preprocess data
dataset_folder_path = '/Users/nanxuan/Desktop/5563/Assignment3/data/THUCNews'
texts, labels = [], []
print("Starting data loading...")

# Iterate over each item in the dataset folder
for folder_name in os.listdir(dataset_folder_path):
    subfolder_path = os.path.join(dataset_folder_path, folder_name)

    # Check if the item is a directory
    if os.path.isdir(subfolder_path):
        # Iterate over each file in the subdirectory
        for file_name in os.listdir(subfolder_path):
            if file_name.endswith('.txt'):
                label = folder_name  # Use the folder name as the label
                file_path = os.path.join(subfolder_path, file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    texts.append(file.read())
                    labels.append(label)
                print(f"Loaded {label} from {file_name}")

print("Data loading complete. Starting preprocessing...")

# Tokenize the text and prepare training data
print("Tokenizing texts...")
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=200)  # Assuming max length of text is 100
print("Shape of data (input features):", data.shape)

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

unique_labels, counts = np.unique(encoded_labels, return_counts=True)
print("Distribution of class labels:")
for label, count in zip(unique_labels, counts):
    print(f"Class {label_encoder.inverse_transform([label])[0]}: {count} samples")


# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Testing labels shape:", y_test.shape)

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
#model = Word2Vec.load("/Users/nanxuan/Desktop/5563/Assignment3/word2vec3.model")
#my_word2vec_model = model.wv

# Get embedding matrices
print("Getting embedding matrices...")
w2v_embedding_matrix = get_embedding_matrix(word_index, pretrained_w2v, 128, False)
glove_embedding_matrix = get_embedding_matrix(word_index, pretrained_glove, 300, False)
#self_trained_w2v_matrix = get_embedding_matrix(word_index, my_word2vec_model, 40, True)
print("Shape of Word2Vec embedding matrix:", w2v_embedding_matrix.shape)

# Function to build and train the model
def build_and_train_model(X_train, y_train, X_test, y_test, embedding_matrix):
    embedding_dim = embedding_matrix.shape[1]
    model = Sequential()
    model.add(
        Embedding(input_dim=embedding_matrix.shape[0], output_dim=128, weights=[embedding_matrix], input_length=200,
                  trainable=True))
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(50)))
    model.add(Dropout(0.3))
    model.add(Dense(output_units, activation='softmax'))
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=64, callbacks=[early_stopping])
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
#print_model_performance(model_self_w2v, X_train, y_train, X_test, y_test)

# Train the Word2Vec model and print its performance
model_w2v = build_and_train_model(X_train, y_train, X_test, y_test, w2v_embedding_matrix)
print("Results for Word2Vec model:")
print_model_performance(model_w2v, X_train, y_train, X_test, y_test)

# Train the GloVe model and print its performance
model_glove = build_and_train_model(X_train, y_train, X_test, y_test, glove_embedding_matrix)
print("Results for GloVe model:")
print_model_performance(model_glove, X_train, y_train, X_test, y_test)