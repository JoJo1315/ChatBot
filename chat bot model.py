# This is the actual Chatbot. Still need to fix some issues. When the user inputs a greeting the bot will sometimes give a goodbye response 


import random 
import json 
import pickle
import nltk 
from nltk.stem import WordNetLemmatizer

# import required modules
import tensorflow as tf
import numpy as np
from keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from keras.layers import GlobalMaxPooling2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.models import Sequential, load_model




lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

with open('words.pkl', 'rb') as f:
    words = pickle.load(f)

with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)


model = load_model('chatbot_model.h5')

print(classes)


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence): 
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.6
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse = True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['response'])
            break 
    return result 

print("Go! Bot is running!")

while True:
    message = input("You: ")  # Add a prompt to indicate user input
    ints = predict_class(message)
    print("Predicted Intents:", ints)  # Print the predicted intents
    res = get_response(ints, intents)
    print("Bot:", res)  # Print the selected response
    


    

 
