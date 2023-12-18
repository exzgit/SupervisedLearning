import warnings
warnings.filterwarnings("ignore")
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random
from keras.models import load_model
import discord
from discord.ext import commands



BotAPI = "GUNAKAN BOT TOKENMU SENDIRI"

# Membuat objek Intents
intents = discord.Intents.default()
intents.message_content = True
# Membuat objek bot dengan command prefix 'p/' dan intents yang sudah didefinisikan
bot = commands.Bot(command_prefix='p/', intents=intents)
X_username = ""


# Import Train Bot corpus file for pre-processing
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']
data_file = open("dataset/json/intents.json").read()
intents = json.loads(data_file)



for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add documents to the corpus
        documents.append((w, intent['tag']))
        # Add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print("~This is words list~")
print(words[3:5])
print("-" * 50)
print("~This is documents list~")
print(documents[3:5])
print("-" * 50)
print("~This is classes list~")
print(classes[3:10])

# Lemmatize, lower each word, and remove duplicates
lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# Sort classes
classes = sorted(list(set(classes)))
# Documents = combination between patterns and intents
print("~Document Length~")
print(len(documents), "documents\n\n")
print("-" * 100)
# Classes = intents
print("~Class Length~")
print(len(classes), "classes\n\n", classes)
print("-" * 100)
# Words = all words, vocabulary
print("~Word Length~")
print(len(words), "unique lemmatized words\n\n", words)
# Creating a pickle file to store the Python objects which we will use while predicting
pickle.dump(words, open('data-training/words.pkl', 'wb'))
pickle.dump(classes, open('data-training/classes.pkl', 'wb'))

# Create our training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])

# Convert training list to numpy array
training = np.array(training, dtype=object)

# Separate features (X) and labels (y)
X = np.array(training[:, 0].tolist())
y = np.array(training[:, 1].tolist())

print("Training data created")

# Create NN model to predict the responses
model = Sequential()
model.add(Dense(128, input_shape=(len(X[0]),), activation='relu'))  # Fix input shape
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y[0]), activation='softmax'))  # Fix output size
# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = tf.keras.optimizers.legacy.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fitting and saving the model
hist = model.fit(X, y, epochs=200, batch_size=16, verbose=2)  # Use X and y directly
model.save('/model/LSTMmodels.h5', hist)  # We will pickle this model to use in the future
print("\n")
print("*" * 50)
print("\nModel Created Successfully!")

def clean_up_sentence(sentence):
    # Tokenize the pattern, split words into an array
    sentence_words = nltk.word_tokenize(sentence)
    # Stem each word, create a short form for the word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    # Tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # Bag of words matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # Assign 1 if the current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model, error_threshold=0.25):
    # Filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    
    # Check if any prediction is above the error threshold
    if np.any(res > error_threshold):
        results = [[i, r] for i, r in enumerate(res)]
        # Sort by the strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        return return_list
    else:
        # Return a default intent or handle the case where no intent is above the threshold
        return [{"intent": "default", "probability": "1.0"}]


# Function to get the response from the model
def get_response(ints, intents_json):
    if ints:
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                return result
    return "I'm sorry, I didn't understand that."

def chatbot_response(text):
    ints = predict_class(text, model)
    res = get_response(ints, intents)
    return res

def start_chat():
    print("This is Bot! Your Personal Assistant.\n\n")
    

@bot.event
async def on_ready():
    print(f'Bot is online! Logged in as {bot.user.name}')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Process the message and get the bot response
    user_input = message.content
    global X_username  # Menggunakan variabel global untuk menyimpan nilai
    X_username = message.author.name
    bot_currentResponse = chatbot_response(user_input)

    # Ganti teks {X_username} dengan username pengguna

    bot_response = f"{bot_currentResponse}"
    keluaran = bot_response.replace("<user>",X_username)

    botX = discord.Embed(title=f"Answer to {X_username}", description=keluaran)
    await message.channel.send(embed=botX)


import matplotlib.pyplot as plt





loss = hist.history['loss']
accuracy = hist.history['accuracy']

epochs = range(1, len(loss) + 1)

# Candlestick chart for loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Candlestick chart for accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, accuracy, 'ro', label='Training accuracy')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Save the figure if needed
plt.savefig('history_PNG/training_history.png')

bot.run(BotAPI)
