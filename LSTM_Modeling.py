from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import numpy as np
import keras.utils as ku 
from sklearn.model_selection import train_test_split
import re
import random
import matplotlib.pyplot as plt
from keras.optimizers import Adam

# Data
document = "dataset/txt/NEODATASET.txt"

with open(document, 'r', encoding='utf-8') as doc:
    data = doc.read().splitlines()

# Langkah Pre-processing
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d[a-zA-Z0-9.,?!@#$%^&*(/-=+)]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text

cleaned_data = [clean_text(line) for line in data]

# Tokenisasi data
tokenizer = Tokenizer()

def dataset_preparation(data):
    tokenizer.fit_on_texts(data)
    total_words = len(tokenizer.word_index) + 1
    
    input_sequences = []
    for line in data:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
            
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    
    predictors, label = input_sequences[:,:-1], input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)
    
    return predictors, label, max_sequence_len, total_words

predictors, label, max_sequence_len, total_words = dataset_preparation(cleaned_data)

# Pemisahan data latih dan data uji
predictors_train, predictors_test, label_train, label_test = train_test_split(predictors, label, test_size=0.3, random_state=42)

def create_model(predictors, label, max_sequence_len, total_words):
    model = Sequential()
    model.add(Embedding(total_words, 15))  # Hapus argumen input_length
    model.add(LSTM(6))
    model.add(Dense(total_words, activation='softmax'))
    optimizer = Adam(learning_rate=1.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    history = model.fit(predictors, label, epochs=10, validation_split=0.7, callbacks=[early_stopping], verbose=2)

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot()
    plt.savefig('history_PNG/training_LSTM.png')

    model.save('model/LSTMmodels.h5')

    print("Model saved to disk.")
    
    return model
    
# Buat model baru jika tidak ada model yang tersimpan
model = create_model(predictors, label, max_sequence_len, total_words)

# Fungsi untuk generate teks
def generate_text(seed_text, model, max_sequence_len):
    generated_text = seed_text  # Mulai dengan seed_text yang diberikan
    next_words = random.randrange(50, 250)
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probabilities = model.predict(token_list, batch_size=2, verbose=2)

        # Pilih indeks berdasarkan distribusi probabilitas, dengan elemen acak
        predicted_probabilities = predicted_probabilities / np.sum(predicted_probabilities)
        predicted_index = np.random.choice(len(predicted_probabilities[0]), p=predicted_probabilities[0])

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        generated_text += " " + output_word  # Tambahkan output_word ke hasil yang sudah ada

        seed_text = generated_text  # Update seed_text untuk prediksi selanjutnya

    return generated_text.strip()

# Contoh penggunaan generate_text
while True:
    seed_text = input(">> USER: ")
    generated_text = generate_text(seed_text, model=model, max_sequence_len=max_sequence_len)
    print(generated_text)
