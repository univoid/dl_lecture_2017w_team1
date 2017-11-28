# coding:utf-8
from requests_oauthlib import OAuth1Session
import json
import re
import codecs

oauth_key_dict = {
    "consumer_key":"pl8LbqC447CLv4EHX2zXLJGMZ",
    "consumer_secret":"fWJOS9laj4sriuljLAJW4kdDwMLwsUyK7zPMMejeBARA8MM4zs",
    "access_token":"930696231063556097-dI3oo5VhduscfrpJBsWtVtcrXYxdgT0",
    "access_token_secret":"NObwlxV8XJMHBVN0YPa29SeyYYoJ1BfOlPNF7z4XIWN47"
}

oauth = OAuth1Session(
    oauth_key_dict["consumer_key"],
    oauth_key_dict["consumer_secret"],
    oauth_key_dict["access_token"],
    oauth_key_dict["access_token_secret"]
)

def get_timeline():
    url="https://api.twitter.com/1.1/statuses/user_timeline.json"
    params={
        "screen_name":"gakuishida",
        "count":200,
        "exclude_replies":True,
        "include_rts":False,
        "trim_user":True,
        "tweet_mode":"extended"
    }
    response = oauth.get(url, params = params)

    if response.status_code != 200:
        message = "Failed(%d)" % response.status_code
        print(message)
    
    tweet_list=[]
    tweets = json.loads(response.text)
    for tweet in tweets:
        if u"おはようございます。みなさんの" in tweet["full_text"]:
            tweet["full_text"] = re.sub(r"\n", "", tweet["full_text"])
            tweet_list.append(tweet["full_text"])
    max_id = tweets[-1]["id"]
    # print "max_id:" + str(max_id)

    for i in range(40):
        params = {
            "screen_name": "gakuishida",
            "count": 200,
            "exclude_replies": True,
            "include_rts": False,
            "trim_user": True,
            "max_id": max_id,
            "tweet_mode": "extended"
            }
        response = oauth.get(url, params=params)
        tweets = json.loads(response.text)
        for tweet in tweets:
            if u"おはようございます。みなさんの" in tweet["full_text"]:
                tweet["full_text"] = re.sub(r"\n", "", tweet["full_text"])
                print tweet["created_at"]
                tweet_list.append(tweet["full_text"])
        # ツイートIDの取得
        if tweets != []:
            max_id = tweets[-1]["id"]
            # print "max_id:" + str(max_id)
    f=codecs.open("./save/raw_data.txt","w", "utf-8")
    f.write(u"\n".join(tweet_list))
    f.close()
    return tweet_list

if __name__ == "__main__":
    print(len(get_timeline()))
