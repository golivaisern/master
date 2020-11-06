

import logging
import sys
import csv
import pandas as pd

from configparser import ConfigParser
from pymongo import MongoClient
from twython import Twython, TwythonRateLimitError
from pandas import DataFrame, read_csv

## Se prepara el log: logging.basicConfig(filename='TwitterSNA.log',level=logging.DEBUG)
logging.basicConfig(filename='TwitterSNA.log',level=logging.DEBUG)
logging.info('INICIO')


def get_search_tweets():
    '''
    $ curl --request GET
 --url 'https://api.twitter.com/1.1/search/tweets.json?q=nasa&result_type=popular'
 --header 'authorization: OAuth oauth_consumer_key="consumer-key-for-app",
 oauth_nonce="generated-nonce", oauth_signature="generated-signature",
 oauth_signature_method="HMAC-SHA1", oauth_timestamp="generated-timestamp",
 oauth_token="access-token-for-authed-user", oauth_version="1.0"'
$ twurl /1.1/search/tweets.json?q=nasa&result_type=popular

    :return:
    '''

    pass


