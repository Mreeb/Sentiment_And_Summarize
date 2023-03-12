import csv
import tkinter as tk
from tkinter import ttk
from transformers import BertTokenizer, TFBertForSequenceClassification
from tkinter import messagebox
import nltk
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import heapq
from keras.models import load_model
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# model = load_model('my_model.h5')
data = pd.read_csv('new_tweet_emotions.csv')

data = data.dropna()
data = data[data['content'].apply(lambda x: isinstance(x, str))]

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=13)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
labels = ['empty', 'sadness', 'enthusiasm', 'neutral', 'worry', 'surprise', 'love', 'fun', 'hate', 'happiness', 'boredom', 'relief', 'anger']

def predict_label():
    input_text = text_entry.get('1.0', 'end-1c')

    input_encodings = tokenizer(input_text, truncation=True, padding=True, return_tensors='tf')
    prediction = model(input_encodings)[0].numpy()
    predicted_label = labels[prediction.argmax()]

    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(data['content'])
    # model = load_model('my_model.h5')
    # max_len = 100
    # new_sentence = "i love you so much"
    # new_seq = tokenizer.texts_to_sequences([new_sentence])
    # new_pad = pad_sequences(new_seq, maxlen=max_len, padding='post')
    # new_pred = model.predict(new_pad)[0]
    #
    # # Convert the predicted sentiment probabilities into a sentiment label
    # sentiment_labels = data['sentiment']
    # new_sentiment = sentiment_labels[np.argmax(new_pred)]
    #
    # label_map = {'empty': 'empty', 'sadness': 'sadness', 'enthusiasm': 'enthusiasm', 'neutral': 'neutral',
    #              'worry': 'worry', 'surprise': 'surprise', 'love': 'love', 'fun': 'fun', 'hate': 'hate',
    #              'happiness': 'happiness', 'boredom': 'boredom', 'relief': 'relief', 'anger': 'anger'}
    # print(f"The predicted sentiment for '{new_sentence}' is: {label_map[new_sentiment]}")

    # confirm = tk.messagebox.askyesno('Confirm Prediction', f'The predicted label is "{label_map[new_sentiment]}". Is this correct?')
    confirm = tk.messagebox.askyesno('Confirm Prediction', f'The predicted label is "{predicted_label}". Is this correct?')

    if confirm:
        text_entry.delete('1.0', tk.END)
    else:
        label_selection = tk.StringVar(value=labels[0])
        label_dropdown = ttk.Combobox(window, textvariable=label_selection, values=labels, width=30, font=('Helvetica', 16), state='readonly', justify='center')
        label_dropdown.pack(pady=20)

        def save_label():
            correct_label = label_selection.get()
            with open('data.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([input_text, correct_label])
            label_dropdown.destroy()
            text_entry.delete('1.0', tk.END)

        submit_button = tk.Button(window, text='Confirm Emotion', command=save_label, font=('Helvetica', 18))
        submit_button.pack(pady=20)

def summarize_text():
    text = text_entry.get('1.0', 'end-1c')
    sentences = sent_tokenize(text)

    stop_words = set(stopwords.words('english'))

    word_frequencies = {}

    for word in nltk.word_tokenize(text):
        if word not in stop_words:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequency = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequency)

    sentence_scores = {}

    for sent in sentences:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    summary_sentences = heapq.nlargest(int(len(sentences) * 0.33), sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    print(summary)

    text_entry.delete('1.0', tk.END)
    text_entry.insert('1.0', summary)

window = tk.Tk()
window.title('Sentiment Analysis')
window.configure(bg='black')

screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
window.geometry(f'{screen_width}x{screen_height}+0+0')

title_label = tk.Label(window, text='Sentiment Analysis', font=('Helvetica', 30), fg='white', bg='black')
title_label.pack(pady=50)

text_entry = tk.Text(window, height=10, width=50, font=('Helvetica', 18), fg='black', bg='white')
text_entry.pack()

button_frame = tk.Frame(window, bg='black')
button_frame.pack(pady=50)

submit_button = tk.Button(button_frame, text='Submit', command=predict_label, font=('Helvetica', 18), bg='white', fg='black', width=20)
submit_button.pack(side='left', padx=50)

summary_button = tk.Button(button_frame, text='Summary', command=summarize_text, font=('Helvetica', 18), bg='white', fg='black', width=20)
summary_button.pack(side='right', padx=50)

window.mainloop()
