import requests
import functools

def get_result(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        error = data['error']

        if error:
            print("API Error:", error)
            return None

        currency_pairs = data['result']
        return currency_pairs

    except requests.exceptions.RequestException as e:
        print("Request Error:", e)
        return None

def flatten_lists(list_of_lists):
    return functools.reduce(lambda x,y: x+y, list_of_lists)

def vectorize_ticker_stream(ticker=[]):
    """
    Until every pair has had an example processed, we won't be able 
    to vectorize the whole thing. This implies that we must throw out 
    some early samples until each pair has been encountered at least once.
    """
    

    all_pairs = set()
    for item in ticker:
        time_received = list(item.keys())[0]
        pair = item[time_received]['pair']
        all_pairs.add(pair)

    pairs_so_far = set()
    last_known_pair_info = {}
    for i, item in enumerate(ticker):
        time_received = list(item.keys())[0]
        pair = item[time_received]['pair']
        pairs_so_far.add(pair)

        data = item[time_received]['data']
        values = data.values()
        flattened_data = flatten_lists(values)
        last_known_pair_info[pair] = flattened_data

        if not all_pairs.difference(pairs_so_far):
            break

    all_pairs = sorted(all_pairs) # enforce ordering of data
    # by this point last_known_pair_info has every pair populated.
    vectorized_tickers = []
    for item in ticker[i:]:
        time_received = list(item.keys())[0]
        pair = item[time_received]['pair']
        data = item[time_received]['data']
        values = data.values()
        flattened_data = flatten_lists(values)
        last_known_pair_info[pair] = flattened_data

        vectorized_data = [time_received]
        for pair in all_pairs:
            flattened_data = last_known_pair_info[pair]
            vectorized_data.extend(flattened_data)
        vectorized_tickers.append(vectorized_data)
    
    return vectorized_tickers, all_pairs
