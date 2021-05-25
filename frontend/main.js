let url = 'http://127.0.0.1:5000'
let req_params = { credentials: 'include', mode: 'cors', headers: { 'Content-Type': 'application/json' } }

fetch(url + '/username', req_params).then(response => response.json()).then(data => {
    if (data.error == 'user_not_logged') {

        document.getElementById('login-form').style.display = 'block';
        document.getElementById('tweets-view').style.display = 'none';

        fetch(url + '/auth', req_params).then(response => response.json()).then(data => {
            document.getElementById('validation-code').innerHTML = "<a target='_blank' rel='noopener noreferrer' href='" + data.url + "'>url</a>"

        })

    } else {
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

function get_max_class(reply) {

}

function analyse_tweet(x) {
    let tweetIdInput = document.getElementById("tweet-id-input").value

    let tweetId = ''
    let userParam = ''

    if (tweetIdInput) {
        tweetId = tweetIdInput
        userParam = 'other'
    } else {
        let selectedTweet = document.getElementsByClassName("selected")[0]
        tweetId = selectedTweet.id
        userParam = 'same'
    }

    fetch(url + '/retrive_bullying_replies?tweet_id=' + tweetId + '&name_type=' + userParam, req_params).then(response => response.json()).then(data => {
        let table = document.getElementById("replies")
        table.innerHTML = ""

        if (data.replies != "empty") {

            for (let i = 0; i < data.replies.length; i++) {
                let reply = data.replies[i]

                let bullying_class = get_max_class(reply) ? "bullying-reply" : "no-bullying-reply"

                table.innerHTML += "<tr class='" + bullying_class + "'><th>" + (i + 1) + "</th><td>" + reply.author + "</td><td>" + reply.text + "</td><td>" + reply.date + "</td><td>" + reply.SVM + "</td><td>" + reply.RNN + "</td><td>" + reply.BERT + "</td></tr>"
            }
        } else {
            table.innerHTML += "No replies for these tweet !"
        }
    })
}

function highlight(x) {
    let tableElements = document.getElementById("tweets").children
    for (let i = 0; i < tableElements.length; i++) {
        tableElements[i].style.background = '#FFFFFF'
        tableElements[i].classList.remove("selected")
    }
    x.style.background = '#0D6EFD'
    x.classList.add("selected")
}

function get_max_class(reply) {
    let predictions = [reply.SVM, reply.RNN, reply.BERT]
    let occ = predictions.reduce((acc, curr) => (acc[curr] = (acc[curr] || 0) + 1, acc), {})
    return (occ[0] > occ[1] ? 0 : 1)
}