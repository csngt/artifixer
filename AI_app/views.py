from django.shortcuts import render

def index_page(request):
    return render(request, "index.html")


def about_page(request):
    return render(request, "about.html")


def contact_page(request):
    return render(request, "contact.html")

from django.http import HttpResponse
from django.shortcuts import render

import joblib
import numpy as np
from transformers import BertTokenizer
from django.http import HttpResponse
from django.shortcuts import render

model_filename = 'random_forest_model.pkl'
random_forest_model = joblib.load(model_filename)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(X):
    tokens = tokenizer(
        text=list(X),
        add_special_tokens=True,
        max_length=100,
        truncation=True,
        padding='max_length',
        return_tensors='tf',
        return_attention_mask=True,
    )
    return tokens

def ai_detection_page(request):
    if request.method == "POST":
        content_input = request.POST.get("content_input", "")

        if not content_input:
            return HttpResponse("<script>alert('Please enter some content to analyze.');window.location.href='/ai_detection_page/';</script>")

        tokens = tokenize([content_input])
        X_new_features = tokens['input_ids']

        X_new_flat = X_new_features.numpy()

        y_pred_new = random_forest_model.predict(X_new_flat)
        prob_new = random_forest_model.predict_proba(X_new_flat)

        if y_pred_new[0] == 1:
            ai_percentage = prob_new[0][1] * 100
            message = f"AI-generated content detected! AI-likeness: {ai_percentage:.2f}%"
        else:
            ai_percentage = prob_new[0][0] * 100
            ai_percentage = 100 - ai_percentage
            message = f"Content appears to be human-generated. AI-likeness: {ai_percentage:.2f}%"

        return HttpResponse(
            f"<script>alert('{message}');window.location.href='/ai_detection_page/';</script>"
        )

    return render(request, "ai_check.html")



import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps
import numpy as np
from django.http import HttpResponse
from django.shortcuts import render
import cv2 as cv
model = tf.keras.models.load_model("AIGeneratedModel_best.h5")


img_size = 48

def preprocess_image(image_path):

    img = cv.imread(image_path)  
    if img is None:
        print("Error: Unable to load image.")
        return None
    
    img = cv.resize(img, (img_size, img_size)) 
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)  
    return img

def ai_detection_image_page(request):
    if request.method == "POST":
        if 'image_input' not in request.FILES:
            return HttpResponse("<script>alert('No image uploaded. Please try again.');window.location.href='/ai_detection_image_page/';</script>")
        print("start")
        uploaded_image = request.FILES['image_input']

  
        image = Image.open(uploaded_image)
        preprocessed_image = preprocess_image(image)
        print("start PREPROCESS")
        prediction = model.predict(preprocessed_image)
        prediction_value = prediction[0][0] 

        if prediction_value <= 0.5:
            message = f"The uploaded image is Real. "
        else:
            message = f"The uploaded image is AI Generated. "
        print(message)
        return HttpResponse(
            f"<script>alert('{message}');window.location.href='/ai_detection_image_page/';</script>"
        )


    return render(request, "ai_image_detection.html")




from django.shortcuts import render
from django.http import HttpResponse

import nltk
from textblob import TextBlob
import re
from nltk.corpus import wordnet

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def replace_word(word, pos):
    synonyms = []
    for syn in wordnet.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            if lemma.name().lower() != word.lower() and lemma.name().isalpha():
                synonyms.append(lemma.name())
    
    if synonyms:
        common_synonyms = [syn for syn in synonyms if syn not in ["indolent", "faineant", "domestic_dog", "brownness"]]
        if common_synonyms:
            synonyms = common_synonyms
        synonyms = sorted(synonyms, key=lambda x: nltk.FreqDist(synonyms)[x], reverse=True)
        most_common_synonym = synonyms[0]
        return most_common_synonym
    else:
        return word

def clean_symbols(humanized_text):
    humanized_text = re.sub(r'\s+', ' ', humanized_text)
    humanized_text = re.sub(r'\s([^\w\s])', r'\1', humanized_text)
    humanized_text = re.sub(r'([^\w\s])\s', r'\1', humanized_text)
    return humanized_text

def correct_contractions(text):
    contractions = {
        "I'ca n't": "I can't",
        "I' ll": "I'll",
        "she' ll": "she'll",
        "they' re": "they're",
        "we' re": "we're",
        "I' ve": "I've"
    }
    for contraction, corrected in contractions.items():
        text = text.replace(contraction, corrected)
    return text

def humanize_text(text):
    newline_placeholder = "庄周"
    text = text.replace('\n', newline_placeholder)

    sentences = re.findall(r'[^.!?]+[.!?]*|[^\w\s]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    humanized_sentences = []
    for sentence in sentences:
        words = re.findall(r'\w+|[^\w\s]+', sentence)

        try:
            tags = TextBlob(sentence).tags
        except Exception as e:
            continue

        humanized_words = []
        for word, tag in tags:
            if tag.startswith('JJ'):
                humanized_word = replace_word(word, 'a')
                humanized_words.append(humanized_word)
            elif tag.startswith('RB'):
                humanized_word = replace_word(word, 'r')
                humanized_words.append(humanized_word)
            elif tag.startswith('VB'):
                humanized_word = replace_word(word, 'v')
                humanized_words.append(humanized_word)
            elif tag.startswith('NN'):
                humanized_word = replace_word(word, 'n')
                humanized_words.append(humanized_word)
            else:
                humanized_words.append(word)

        humanized_sentence = ''
        word_idx = 0
        for word in words:
            if word.isalpha() and word_idx < len(humanized_words):
                humanized_sentence += humanized_words[word_idx] + ' '
                word_idx += 1
            else:
                humanized_sentence += word + ' '

        humanized_sentence = humanized_sentence.strip()

        if humanized_sentence and not humanized_sentence.endswith(('.', '!', '?')):
            humanized_sentence += '.'

        humanized_sentences.append(humanized_sentence)

    humanized_text = ' '.join(humanized_sentences)

    humanized_text = correct_contractions(humanized_text)

    humanized_text = clean_symbols(humanized_text)
    humanized_text = humanized_text.replace(newline_placeholder, '\n')

    return humanized_text.strip()


def humanize_page(request):
    humanized_text = None
    
    if request.method == "POST":

        input_text = request.POST.get('text_input', '')
        print("///////////////////")
        print(input_text)
        print("///////////////////")
        if input_text:
  
            humanized_text = humanize_text(input_text)
    
    return render(request, 'humanize.html', {'humanized_text': humanized_text})