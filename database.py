import json

def get_ticker(filename):
    ticker = []
    with open(filename, 'r') as fp:
        ticker = fp.read().split('\n')
        ticker = [json.loads(pair) for pair in ticker]
    return ticker
