from flask import Flask,render_template,flash,redirect,request,send_from_directory,url_for, send_file, jsonify
import mysql.connector, os
import os
import torch
import re
from transformers import XLNetTokenizer, XLNetForSequenceClassification
import shap
import numpy as np


# Load the saved model and tokenizer
model_path = './saved_model'
tokenizer = XLNetTokenizer.from_pretrained(model_path)
model = XLNetForSequenceClassification.from_pretrained(model_path)

# Ensure the model is in evaluation mode
model.eval()

# Define a function to preprocess the input text

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove all non-word characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.lower()  # Convert to lowercase
    return text

# Define a function to predict the label of a single input text
def predict(text):
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    
    # Tokenize the text
    inputs = tokenizer(preprocessed_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        
    # Map the predicted class id to the label
    label_map = {0: 'Fake', 1: 'True'}
    predicted_label = label_map[predicted_class_id]
    
    return predicted_label

app = Flask(__name__)

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3306",
    database='test'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data


@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
                values = (name, email, password)
                executionquery(query, values)
                return render_template('index.html', message="Successfully Registered!")
            return render_template('index.html', message="This email ID is already exists!")
        return render_template('index.html', message="Conform password is not match!")
    return render_template('index.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])
        
        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password.upper() == password__data[0][0]:
                global user_email
                user_email = email

                return redirect("/home")
            return render_template('index.html', message= "Invalid Password!!")
        return render_template('index.html', message= "This email ID does not exist!")
    return render_template('index.html')


@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/prediction', methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        text = request.form['text']
        predicted_label = predict(text)
        
        return render_template('home.html', prediction = predicted_label,text=text)
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)

