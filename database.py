import json

from utilities import flatten_lists

def get_ticker(filename):
    ticker = []
    with open(filename, 'r') as fp:
        ticker = fp.read().split('\n')
        ticker = [json.loads(pair) for pair in ticker]
    return ticker

def vectorize_ticker(ticker=[]):
    last_known_pair_info = {}

    for pair in ticker:
        time_received = list(pair.keys())[0]
        data = pair[time_received]['data'].values()
        flattened_data = flatten_lists(data)
