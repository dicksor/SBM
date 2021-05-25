import settings
import tweepy
import secrets
import numpy as np

from flask import Flask, request, session
from flask_cors import CORS, cross_origin

app = Flask(__name__)

app.secret_key = secrets.token_hex(16)
app.config['JSON_AS_ASCII'] = False
app.config['CORS_HEADERS'] = 'Content-Type'
app.config.update(SESSION_COOKIE_SAMESITE="None", SESSION_COOKIE_SECURE=True)


SESSION_TYPE = 'filesystem'

CORS(app)
cors = CORS(app, support_credentials=True)

def get_auth_object():
    if 'token' in session:
        token, token_secret = session['token']
        auth = tweepy.OAuthHandler(settings.API_KEY, settings.API_SECRET_KEY)
        auth.set_access_token(token, token_secret)

        return auth
    else:
        return None

def get_predictions(replies):
    '''
    return (SVM, RNN, BERT)
    '''
    return (np.ones(len(replies)), np.ones(len(replies)), np.ones(len(replies)))


def get_replies(api, name, tweet_id):
    '''
    return (author, text, date)
    '''
    replies = []
    for tweet in tweepy.Cursor(api.search,q='to:'+name, result_type='recent').items(100):
        if hasattr(tweet, 'in_reply_to_status_id_str'):
            if (tweet.in_reply_to_status_id_str==tweet_id):
                replies.append((tweet.author.screen_name, tweet.text, tweet.created_at))

    print(replies)
    return np.array(replies)

@app.route('/auth', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_auth_url():
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
    auth = get_auth_object()
    if auth is not None:
        return {'name' : tweepy.API(auth).me().screen_name}
    else:
        return {'error':'user_not_logged'}

@app.route('/user_tweets')
@cross_origin(supports_credentials=True)
def user_tweets():
    auth = get_auth_object()
    if auth is not None:
        result = {"tweets":[]}
        for tweet in tweepy.API(auth).me().timeline():
            result["tweets"].append({"id":tweet.id, "text":tweet.text, "date":tweet.created_at})
        
        return result
    else:
        return {'error':'User is not auth !'}

@app.route('/retrive_bullying_replies')
@cross_origin(supports_credentials=True)
def retrive_bullying_tweets():
    if request.args.get('tweet_id') is not None and request.args.get('name_type') is not None:
        tweet_id = request.args.get('tweet_id')
        name_type = request.args.get('name_type')

        auth = get_auth_object()
        if auth is not None:
            api = tweepy.API(auth)
            name = ''

            if name_type == 'same' : # User want analyse hos own tweet
                name = tweepy.API(auth).me().screen_name
            elif name_type == 'other' : # User want analyse specific tweet
                tweet = api.get_status(tweet_id)
                name = tweet.user.name
            else : 
                return {'error':'Arg not authorized ! '}

            print(name)
            print(tweet_id)
            replies = get_replies(api, name, tweet_id)

            if len(replies) >= 1:
                _, text, _ = zip(*replies)
                predictions = get_predictions(text)

                formatted_prediction = []
                for row in zip(replies, *predictions):
                    rep = row[0]
                    pred = row[1:4]
                    formatted_prediction.append({"author":rep[0], "text":rep[1], "date":rep[2], "SVM":pred[0], "RNN":pred[1], "BERT":pred[2]})

                return {"replies":formatted_prediction}
            else :
                return {"replies":"empty"}
        else:
            return {'error':'User is not auth !'}
    else:
         return {'error':'Params not defined in the request'}

@app.route('/')
@cross_origin(supports_credentials=True)
def default():
    return 'backend'