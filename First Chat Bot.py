

#For training the chatbot
import random 
import json 
import pickle
import numpy as np 

import nltk 
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
#Reduce word to its stem 

import tensorflow as tf

lemmatizer = WordNetLemmatizer()
 
intents = json.loads(open('intents.json').read())
#reading the json file as text, then loading to the json object. Loads method converts the JSON file to a python dictionary

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

#Iterate over the intents

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        #Splits up sentence into its individual words
        words.extend(word_list) #Adding the tokenized words into the word list
        #Extending means taking the content and adding it to the list 
        #Appening means taking the LIST and adding it to another list
        documents.append((word_list, intent['tag']))
        #So we know that the word list belongs to its respective tag
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            #check if class is alread in classes list. If not then append it to the class list

words = [lemmatizer.lemmatize(word) for word in words if words not in ignore_letters]
#Lemmative the word for ever word in words if this word is not in the ignore letters list
words = sorted(set(words))
#Set eliminates the duplicates and sorted turns it back into a list 
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


#Neural Network needs Numerical values

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    #For every combination we are going to create an empty bag of words
    word_patterns = document[0]
    #Word patterns is what we find in the word document at index 0 
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words: # For each word we want to know if it occurs in word patterns 
        bag.append(1)if word in word_patterns else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
train_x = []
train_y = []

for item in training:
    train_x.append(item[0])
    train_y.append(item[1])

# Creating the Neural Network
model = tf.keras.Sequential()
# Input layer. 128 Neurons. The shape is dependent on the shape of the x axis
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))
# Softmax is the function that will allow us to add up the results in the output layer that it all adds up to 1

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print('Done')