# libraries
import random
import numpy as np
import pickle
import json
from flask import Flask, render_template, request
#from flask_ngrok import run_with_ngrok
import nltk
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
import pandas as pd
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

from sklearn.model_selection import train_test_split
lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

app = Flask(__name__,)

#laoding the intent file
intents = json.loads(open("Intents_Chatbot_eCommerce.json").read())

#creatifng 3 arrays for words, classes, and the documents to loop and append it to each document
words=[]
classes = []
documents = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        #add documents in the corpus
        documents.append((wrds, intent['tag']))

        if intent['tag'] not in classes:
          classes.append(intent['tag'])



ignore_words = ['?', '!']
eng_stop_words = nltk.corpus.stopwords.words('english')
# Preprocessing: Removing stop words, lemmatization, lower each word and remove duplicates
words = [word for word in words if word not in eng_stop_words]
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# sort and adding classes to a list
classes = sorted(list(set(classes)))

# creating pickle files for words and classes array
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

#Reading the dataset 
df = pd.read_excel('FinalTableAmazon.xlsx')

#Preprocessing the number of stocks available 
df['number_available_in_stock'].replace(regex=True, inplace=True, to_replace=r'[^0-9]', value=r'')
df['number_available_in_stock'] = df['number_available_in_stock'].notnull().astype(int)

#df['order_approved_at'] = pd.to_datetime(df["order_approved_at"]).dt.date
df["order_approved_at"] = [d.date() for d in df["order_approved_at"]]
df["order_shipped_date"] = [d.date() for d in df["order_shipped_date"]]
df["order_purchase_timestamp"] = [d.date() for d in df["order_purchase_timestamp"]]
df["order_delivered_customer_date"] = [d.date() for d in df["order_delivered_customer_date"]]
df["order_estimated_delivery_date"] = [d.date() for d in df["order_estimated_delivery_date"]]


# Setting up training data and access to the intent to identify the response pmnce the tag is predicted
training = []
# empty array in position 0 for the output
output_empty = [0] * len(classes)
# On the training set the BOW will be applied on each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # completting lemmatization for each word 
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])


# splitting data into test and train
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.15)

# Create model - 3 layers. First layer 90 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
# model = Sequential()
# model.add(Dense(90, input_shape=(len(train_x[0]),), activation='relu'))
# model.add(Dropout(0.6))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.6))
# model.add(Dense(len(train_y[0]), activation='softmax'))

# # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# #fitting and saving the model 
# hist = model.fit(np.array(X_train), np.array(y_train), epochs=200, batch_size=5, verbose=1,
#                  validation_data=(np.array(X_test), np.array(y_test)))
# model.save('chatbot_model.h5', hist)

# #checking the model accuracy
# score = model.evaluate(np.array(X_test),np.array(y_test), batch_size=5, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

# chat initialization
model = load_model("chatbot_model.h5")
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

# chat functionalities
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.45
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    #print(tag)
    list_of_intents = intents_json['intents']
    #print(list_of_intents)
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            #print(result)
            break
    return result




@app.route("/")
def home():
    return render_template("index.html")


@app.route("/getresponse", methods=["POST"])
def chatbot_response():
  try:

    text = request.form["msg"]
    ints = predict_class(text, model)
      
      #print("class is", ints[0]['intent'])
      #print(ints[0]['intent']=='order_id')

    # if there is no corresponding intent
    if len(ints) == 0:
      out_string = 'Hey Sorry! I''m at learning stage. Please search for something more precise.'
      return out_string
    print("int is-------", ints)

    intent = ints[0]['intent']
      
    # if len(text)>5:
    if ints[0]['intent']=='order_status':
      return order_details(text,intent)
    
    elif ints[0]['intent']=='order_approved':
      return order_details(text,intent)

    elif ints[0]['intent']=='order_shipped':
      return order_details(text,intent)

    elif ints[0]['intent'] =='order_arrival_estimated':
      return order_details(text,intent)

    elif ints[0]['intent'] == 'order_delivered':
      return order_details(text,intent)

    elif ints[0]['intent'] == 'Payment_Type':
      return order_details(text, intent)

    elif ints[0]['intent'] == 'Total_cost_Order':
      return order_details(text, intent)

    elif ints[0]['intent'] == 'Installments':
      return order_details(text, intent)

    elif ints[0]['intent'] == 'Product_shipping_cost':
      return order_details(text, intent)

    elif ints[0]['intent'] == 'product_information':
      return order_details(text, intent)

    elif ints[0]['intent'] == 'Product_price':
      return order_details(text, intent)

    elif ints[0]['intent'] == 'Product_availability':
      return order_details(text, intent)

    else:
      res = getResponse(ints, intents)
      return res
  except:
    print('Something unsual happend, Can you please try again? We apologize for this')


def order_details(inp, intent):
  try:
    #Splitting text to take out the potential ids. 
    input_array = inp.split()
    input_ids = [x for x in input_array if len(x) >= 10]
    input_ids_clean =[''.join(e for e in x if e.isalnum()) for x in input_ids]
    #print(input_ids_clean)


    is_valid_id = False
    actual_id = 0

    # looping through potential ids
    for id in  input_ids_clean: 
      row = df[df['order_id']==id]
      if len(row)==0:
        is_valid_id = False
      else:
        is_valid_id = True
        actual_id = id
        break
    
    row = df[df['order_id']==actual_id]

    if (is_valid_id == False):
      return("I'm sorry! I'm not able to find the order with the id you provided. Can you please check again?")

    else:
      print("order id is:", actual_id)
      if (intent == "order_status"):
        out_string = 'Your order with id '+ actual_id+' was ' + row['order_status'].values[0] +'. Thank you for shopping with us!'
        return out_string

      elif (intent == "order_approved"):
        out_string = 'You purchased the order with order id '+ actual_id +' on ' + str(row['order_purchase_timestamp'].values[0])+' and it was approved on ' + str(row['order_approved_at'].values[0])+ '. Thank you for shopping with us!'
        return out_string

      elif (intent == "order_shipped"):
        out_string = 'Your order with order id '+ actual_id +' was shipped at ' + str(row['order_shipped_date'].values[0])+'. Thank you for shopping with us!'
        return out_string

      elif (intent == "order_arrival_estimated"):
        out_string = 'Your order with order id '+ actual_id +' is estimated to be delivered on ' + str(row['order_estimated_delivery_date'].values[0])+'. Thank you for shopping with us!'
        return out_string

      elif (intent == "order_delivered"):
        out_string = 'Your order with order id '+ actual_id +' was delivered on ' + str(row['order_delivered_customer_date'].values[0])+'. Thank you for shopping with us!'
        return out_string

      elif (intent == "Payment_Type"):
        out_string = 'Your order with order id '+ actual_id +' was paid through ' + row['payment_type'].values[0]+'. Thank you for shopping with us!'
        return out_string

      elif (intent == "Total_cost_order"):
        out_string = 'The total cost of your order with order id '+ actual_id +' is ' + str(row['total order amount'].values[0])+'. Thank you for shopping with us!'
        return out_string

      elif (intent == "Installments"):
        out_string = 'Your total number of installments for order with order id is '+ actual_id +': ' + str(row['payment_installments'].values[0])+'. Thank you for shopping with us!'
        return out_string

      elif (intent == 'Product_shipping_cost'):
        out_string = 'The total cost for your purchase with order id '+ actual_id +' is ' + str(row['total order amount'].values[0])+'. The price of your product is $'+str(row['Purchase price'].values[0])+' and the shipping charges for your purchase is $'+ str(row['Shipping Fee'].values[0]) +'. Thank you for shopping with us!'
        return out_string

      elif (intent == 'Product_price'):
        out_string = 'The cost for your product with order id '+ actual_id +' is $'+str(row['Purchase price'].values[0])+ '. Thank you for shopping with us!'
        return out_string

      elif (intent == 'Product_availability'):
        if (int(row['number_available_in_stock'].values[0]) > 0):
          out_string = 'This product is currently in stock in case you want to buy more. Thank you for shopping with us!'
        else:
          out_string = 'This product is out of stock. Please wait for few days. Thank you for shopping with us!'
        return out_string

      elif (intent == "product_information"):
        out_string = 'The product you bought:' + row['product_name'].values[0] +' -- is manufactured by ' + row['Manufacturer_Name'].values[0] 
        out_string = out_string +  '. The price is $'+str(row['Purchase price'].values[0])+' and the shipping charges for the product are $'+ str(row['Shipping Fee'].values[0])
        out_string = out_string + '.This product has ' + row['average_review_rating'].values[0]+'. '
        
        #print((row['average_review_rating'].values[0]))
        if (int(row['number_available_in_stock'].values[0]) > 0):
          out_string = out_string + ' This product is currently in stock in case you want to buy more. Thank you for shopping with us!'
        else:
          out_string = out_string +' This product is out of stock. Please wait for few days. Thank you for shopping with us!'

        return out_string
  except:
    print('Something unsual happend, Can you please try again? We apologize for this')


if __name__ == "__main__":
    app.run()