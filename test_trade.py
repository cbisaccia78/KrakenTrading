import numpy as np

from utilities import timestamp_to_percent, vectorize_window, vectorize_from_cache, update_pair_cache
from test_utilities import ticker


window_length = 5

raw_ticker_stream = ticker
pair_cache = {}

examples_processed = 0
examples_received = 0

NUM_EXAMPLES = 4459


# update the pair cache for all examples that came in while model was trading / sleeping,
# UNLESS those examples will be apart of the window_length
last_uncached_index = examples_received - window_length
if last_uncached_index - examples_processed > 0:
    update_pair_cache(raw_ticker_stream[examples_processed:last_uncached_index], pair_cache)

# grab the most recent window_length examples
recent_examples = raw_ticker_stream[last_uncached_index:]
# vectorize examples from the cache
recent_examples = vectorize_from_cache(pair_cache, recent_examples) # this call will also update pair cache with recent examples

#convert to numpy array
x = np.array(recent_examples, dtype=np.float32)
x[:, 0] = timestamp_to_percent(x[:, 0])

x = vectorize_window(x, window_length)