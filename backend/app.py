import settings
import tweepy
import secrets

from flask import Flask, request, session

app = Flask(__name__)

app.secret_key = secrets.token_hex(16)
app.config['JSON_AS_ASCII'] = False

def get_auth_object():
    if 'token' in session:
        token, token_secret = session['token']
        auth = tweepy.OAuthHandler(settings.API_KEY, settings.API_SECRET_KEY)
        auth.set_access_token(token, token_secret)

        return auth
    else:
        return None

@app.route('/auth')
def get_auth_url():
    try:
        auth = tweepy.OAuthHandler(settings.API_KEY, settings.API_SECRET_KEY)
        url = auth.get_authorization_url()
        session['request_token'] = auth.request_token

        return url
    except tweepy.TweepError:
        return 'Error! Failed to get request token.'

@app.route('/set_verifier')
def set_access_token():
    if request.args.get('verifier') is not None:
        try : 
            verifier = request.args.get('verifier')

            auth = tweepy.OAuthHandler(settings.API_KEY, settings.API_SECRET_KEY)
            auth.request_token = session['request_token']

            auth.get_access_token(verifier)
            session['token'] = (auth.access_token, auth.access_token_secret)

            return {'status':'success'}

        except tweepy.TweepError as e:
            return str(e)
    else :
        return {'error':'Token not defined in the request'}

@app.route('/user_name')
def user_name():
    auth = get_auth_object()
    if auth is not None:
        return {'name' : tweepy.API(auth).me().screen_name}
    else:
        return {'error':'User is not auth !'}

@app.route('/user_tweets')
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
def retrive_bullying_tweets():
    if request.args.get('tweet_id') is not None:
        tweet_id = request.args.get('tweet_id')

        auth = get_auth_object()
        if auth is not None:
            api = tweepy.API(auth)
            name = ''
            if request.args.get('name') is None : 
                name = tweepy.API(auth).me().screen_name
            else :
                name = request.args.get('name')

            result = {"replies":[]}

            for tweet in tweepy.Cursor(api.search,q='to:'+name, result_type='recent').items(100):
                if hasattr(tweet, 'in_reply_to_status_id_str'):
                    if (tweet.in_reply_to_status_id_str==tweet_id):
                        result["replies"].append({"author":tweet.author.screen_name, "text":tweet.text})

            return result
        else:
            return {'error':'User is not auth !'}
    else:
         return {'error':'Tweet id not defined in the request'}
