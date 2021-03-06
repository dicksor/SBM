import settings
import tweepy
import secrets
import numpy as np
import joblib
import os
import pandas as pd
import json

from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing import text
from flask import Flask, request, session
from flask_cors import CORS, cross_origin

from SBM.TextProcessor import *
from SBM.BERT import *

app = Flask(__name__)

app.secret_key = secrets.token_hex(16)
app.config['JSON_AS_ASCII'] = False
app.config['CORS_HEADERS'] = 'Content-Type'
app.config.update(SESSION_COOKIE_SAMESITE="None", SESSION_COOKIE_SECURE=True)

SESSION_TYPE = 'filesystem'

CORS(app)
cors = CORS(app, support_credentials=True)

# Load SVM model
model_svm = joblib.load(os.path.join('models','tfidf_svc.pkl'))

tp = TextProcessor(remove_punctuation=True, 
                   remove_stop_word=True, 
                   min_word_size=2, 
                   special_token_method=SpecialTokenMethod.PREPROCESS)

# Load GRU model
GRU_MAX_LEN = 40
model_gru = load_model(os.path.join('models','model_final.h5'))

tokenizer = None
with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer_gru = text.tokenizer_from_json(data)

# Load BERT
tokenizer_bert = AutoTokenizer.from_pretrained('bert-base-cased', model_max_length=140)
bert_cls = SBMBertClassifier(n_epochs=5, tokenizer=tokenizer)


def get_auth_object():
    """Get the auth connection object for a user

    Returns:
        tweepy.OAuthHandler: Tweepy OAuthHandler
    """
    if 'token' in session:
        token, token_secret = session['token']
        auth = tweepy.OAuthHandler(settings.API_KEY, settings.API_SECRET_KEY)
        auth.set_access_token(token, token_secret)

        return auth
    else:
        return None

def get_predictions(replies):
    """Get predictions for the model SVM, GRU, BERT

    Args:
        replies (list): List of text replies

    Returns:
        tuple: tuple of shape (SVM pred, GRU pred, BERT pred)
    """
    # SVM
    X_pre = tp.fit_transform(np.array(replies))

    # GRU
    X_pre_gru = tokenizer_gru.texts_to_sequences(X_pre)
    X_pre_gru = pad_sequences(X_pre_gru, padding='post', maxlen=GRU_MAX_LEN)

    # BERT 
    os.path.join('backend', 'models', 'BERT', 'checkpoint-4215')
    y_bert, _ = bert_cls.test(TestDataset(X_pre, tokenizer_bert), os.path.join('models', 'BERT', 'checkpoint-4215'))

    y_svm = model_svm.predict(X_pre)
    y_gru = np.where(model_gru.predict(X_pre_gru) > 0.5, 1, 0)
    y_gru = [int(y) for y in y_gru.flatten()]
    y_bert = [int(y) for y in y_bert]


    return (y_svm, y_gru, y_bert)


def get_replies(api, name, tweet_id):
    """Get a tweet replies

    Args:
        api ([type]): Tweepy API object
        name (string): Username of tweet owner
        tweet_id (string): Tweet id

    Returns:
        np.array: list of tuple of shape (tweet author, tweet text, tweet creation date)
    """
    replies = []
    for tweet in tweepy.Cursor(api.search,q='to:'+name, result_type='recent').items(250):
        if hasattr(tweet, 'in_reply_to_status_id_str'):
            if (tweet.in_reply_to_status_id_str==tweet_id):
                replies.append((tweet.author.screen_name, tweet.text, tweet.created_at))

    return np.array(replies)

@app.route('/auth', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_auth_url():
    """Get autentification url

    Returns:
        flask.Response: Auth url and request token
    """
    try:
        auth = tweepy.OAuthHandler(settings.API_KEY, settings.API_SECRET_KEY)
        url = auth.get_authorization_url()
        session['request_token'] = auth.request_token
        print(session['request_token'])

        return {'url':url, 'request_token':session['request_token']}
    except tweepy.TweepError:
        return 'Error! Failed to get request token.'

@app.route('/set_verifier')
@cross_origin(supports_credentials=True)
def set_access_token():
    """Define user access token

    Returns:
        flask.Response: status
    """
    if request.args.get('verifier') is not None:
        try : 
            verifier = request.args.get('verifier')

            auth = tweepy.OAuthHandler(settings.API_KEY, settings.API_SECRET_KEY)
            print(session['request_token'])
            auth.request_token = session['request_token']

            auth.get_access_token(verifier)
            session['token'] = (auth.access_token, auth.access_token_secret)

            return {'status':'success'}

        except tweepy.TweepError as e:
            return  {'error':str(e)}
    else :
        return {'error':'Token not defined in the request'}

@app.route('/username')
@cross_origin(supports_credentials=True)
def user_name():
    """Get logged username

    Returns:
        flask.Response: Logged username
    """
    auth = get_auth_object()
    if auth is not None:
        return {'name' : tweepy.API(auth).me().screen_name}
    else:
        return {'error':'user_not_logged'}

@app.route('/user_tweets')
@cross_origin(supports_credentials=True)
def user_tweets():
    """Logged user tweets

    Returns:
        flask.Response: List of tuple of shape (tweet id, tweet text, tweet date)
    """
    auth = get_auth_object()
    if auth is not None:
        result = {"tweets":[]}
        for tweet in tweepy.API(auth).me().timeline():
            result["tweets"].append({"id":str(tweet.id), "text":tweet.text, "date":tweet.created_at})
        
        return result
    else:
        return {'error':'User is not auth !'}

@app.route('/retrive_bullying_replies')
@cross_origin(supports_credentials=True)
def retrive_bullying_tweets():
    """Get bullying tweet from an tweet id

    Returns:
        flask.Response: Analysed tweet username and text and for each reply : author, text, date, models predictions, 
    """
    if request.args.get('tweet_id') is not None and request.args.get('name_type') is not None:
        tweet_id = request.args.get('tweet_id')
        name_type = request.args.get('name_type')

        auth = get_auth_object()
        if auth is not None:
            api = tweepy.API(auth)
            name = ''
            tweet_text = ''
            display_name = ''

            if name_type == 'same' : # User want analyse hos own tweet
                name = tweepy.API(auth).me().screen_name
            elif name_type == 'other' : # User want analyse specific tweet
                try : 
                    tweet = api.get_status(tweet_id)
                except :
                    return {'error':'Tweet id not valid ! '}

                name = tweet.user.screen_name
                display_name = tweet.user.name
                tweet_text = tweet.text
            else : 
                return {'error':'Arg not authorized ! '}

            print(name)

            replies = get_replies(api, name, tweet_id)

            if len(replies) >= 1:
                _, text, _ = zip(*replies)
                predictions = get_predictions(text)

                formatted_prediction = []
                for row in zip(replies, *predictions):
                    rep = row[0]
                    pred = row[1:4]
                    formatted_prediction.append({"author":rep[0], "text":rep[1], "date":rep[2], "SVM":pred[0], "RNN":pred[1], "BERT":pred[2]})

                return {"replies":formatted_prediction, "username":display_name, "tweet":tweet_text}
            else :
                return {"replies":"empty"}
        else:
            return {'error':'User is not auth !'}
    else:
         return {'error':'Params not defined in the request'}

@app.route('/')
@cross_origin(supports_credentials=True)
def default():
    """Default root

    Returns:
        flask.Response: 
    """
    return 'backend'