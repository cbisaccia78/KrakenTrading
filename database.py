import json

def get_ticker_stream(filename):
    ticker_stream = []
    with open(filename, 'r') as fp:
        raw_ticker_stream = fp.read().split('\n')
        raw_ticker_stream.pop(-1)
        for pair in raw_ticker_stream:
            try:
                p = json.loads(pair)
                ticker_stream.append(p)
            except Exception as e:
                print(e)
                return []
    return ticker_stream
