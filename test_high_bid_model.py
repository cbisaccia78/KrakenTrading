from high_bid_model import create_model
from database import get_ticker_stream

pair_name = 'EUR/USD'
window_length = 5

raw_ticker_stream = get_ticker_stream('./ticker-1-test')

model, test_mse, standard_scalar, pair_cache = create_model(raw_ticker_stream, pair_name, window_len=window_length)

print(pair_cache)