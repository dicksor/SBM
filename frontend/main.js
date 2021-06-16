let url = 'http://127.0.0.1:5000'
let req_params = { credentials: 'include', mode: 'cors', headers: { 'Content-Type': 'application/json' } }

/*
Get the username of the user
*/
fetch(url + '/username', req_params).then(response => response.json()).then(data => {

    // If user not auth, get url login for the user
    if (data.error == 'user_not_logged') {

        document.getElementById('login-form').style.display = 'block';
        document.getElementById('tweets-view').style.display = 'none';

        fetch(url + '/auth', req_params).then(response => response.json()).then(data => {
            document.getElementById('validation-code').innerHTML = "<a target='_blank' rel='noopener noreferrer' href='" + data.url + "'>url</a>"

        })

    } else { // Otherwise display user tweets
        document.getElementById('login-form').style.display = 'none';
        document.getElementById('tweets-view').style.display = 'block';

        document.getElementById('username').innerHTML = data.name

        fetch(url + '/user_tweets', req_params).then(response => response.json()).then(data => {

            for (let i = 0; i < data.tweets.length; i++) {
                let tweet = data.tweets[i]

                document.getElementById("tweets").innerHTML += "<tr onclick='highlight(this)' id=" + tweet.id + "><th>" + (i + 1) + "</th><td>" + tweet.text + "</td><td>" + tweet.date + "</td></tr>"
            }
        })
    }
});

/*
Function for define the user token in the backend
*/
function set_verifier(x) {
    let validation_token = document.getElementById("validation-code-input").value
    if (validation_token != '') {
        fetch(url + '/set_verifier?verifier=' + validation_token, req_params)
            .then(response => response.json())
            .then(data => {
                window.location.reload();
            })

    } else {
        alert("Please enter a token !")
    }
}

/*
Retrive bullying tweet
*/
function analyse_tweet(x) {
    let tweetIdInput = document.getElementById("tweet-id-input").value

    let tweetId = ''
    let userParam = ''

    // If tweet id is empty get user own tweet
    if (tweetIdInput) {
        tweetId = tweetIdInput
        userParam = 'other'
    } else { // Otherwise get the id in the input field
        let selectedTweet = document.getElementsByClassName("selected")[0]
        tweetId = selectedTweet.id
        userParam = 'same'
    }

    let table = document.getElementById("replies")
    table.innerHTML = ""
    document.getElementById("loader-span").style.display = "block"

    // Get tweets replies from an ID
    fetch(url + '/retrive_bullying_replies?tweet_id=' + tweetId + '&name_type=' + userParam, req_params).then(response => response.json()).then(data => {
        document.getElementById('specifi-tweet-info').innerHTML = ""

        // If not error occur and replies not empty
        if (!data.error) {
            if (data.replies != "empty") {

                if (data.tweet != '') {
                    document.getElementById('specifi-tweet-info').innerHTML += '<p><strong>Username : </strong>' + data.username + '</p><p><strong>Tweet : </strong>' + data.tweet + '</p>'
                }

                for (let i = 0; i < data.replies.length; i++) {
                    let reply = data.replies[i]

                    let bullying_class = get_max_class(reply) ? "bullying-reply" : "no-bullying-reply"

                    table.innerHTML += "<tr class='" + bullying_class + "'><th>" + (i + 1) + "</th><td>" + reply.author + "</td><td>" + reply.text + "</td><td>" + reply.date + "</td><td>" + reply.SVM + "</td><td>" + reply.RNN + "</td><td>" + reply.BERT + "</td></tr>"
                }
            } else {
                table.innerHTML += "No replies for these tweet !"
            }
        } else {
            table.innerHTML += "Tweet id not valid !"
        }

        document.getElementById("loader-span").style.display = "none"
    })
}

/*
Higlight a tweet on click
*/
function highlight(x) {
    let tableElements = document.getElementById("tweets").children
    for (let i = 0; i < tableElements.length; i++) {
        tableElements[i].style.background = '#FFFFFF'
        tableElements[i].classList.remove("selected")
    }
    x.style.background = '#0D6EFD'
    x.classList.add("selected")
}

/*
Unhighlight tweets
*/
function unhighlight() {
    let tableElements = document.getElementById("tweets").children
    for (let i = 0; i < tableElements.length; i++) {
        tableElements[i].style.background = '#FFFFFF'
        tableElements[i].classList.remove("selected")
    }
}

/*
Get the majority class from models predictions
*/
function get_max_class(reply) {
    let predictions = [reply.SVM, reply.RNN, reply.BERT]
    let occ = predictions.reduce((acc, curr) => (acc[curr] = (acc[curr] || 0) + 1, acc), {})
    if (!occ[0]) {
        return 1
    } else if (!occ[1]) {
        return 0
    } else {
        return (occ[0] > occ[1] ? 0 : 1)
    }
}