import websocket
import json
import datetime
import threading

import numpy as np

from api import get_all_wsnames
from high_bid_model import create_model, FEATURE_MAP
from utilities import vectorize_from_cache, timestamp_to_percent, vectorize_window, update_pair_cache

# TODO - move these to api module
PUBLIC_URL = "wss://ws.kraken.com/" 
PRIVATE_URL = "wss://ws-auth.kraken.com/"

ACCEPTED_ERROR = 0.01

features_per_pair = len(FEATURE_MAP)
pair_name = 'XBT/USD'

window_length = 5

model = None
model_thread = None

retrain = False

raw_ticker_stream = []
pair_cache = {}

errors = []

examples_processed = 0
examples_received = 0

NUM_EXAMPLES = 4459

def model_thread_func():

    global model
    global raw_ticker_stream
    global pair_cache
    global examples_processed
    global examples_received
    global retrain
    global errors

    if model is None: # need to create model
            if examples_received >= NUM_EXAMPLES: 
                model, test_mse, standard_scalar, pair_cache = create_model(raw_ticker_stream, pair_name, window_len=window_length)
                examples_processed = NUM_EXAMPLES
    
    elif retrain: # model is not performing well
        examples_received = 0
        examples_processed = 0
        model = None

    else: # we can trade

        # update the pair cache for all examples that came in while model was trading / sleeping,
        # UNLESS those examples will be apart of the window_length
        last_uncached_index = examples_received - window_length
        if last_uncached_index - examples_processed > 0:
            update_pair_cache(raw_ticker_stream[examples_processed:last_uncached_index], pair_cache)

        # grab the most recent window_length examples
        example_window = raw_ticker_stream[last_uncached_index:]
        # vectorize examples from the cache
        example_window = vectorize_from_cache(pair_cache, example_window) # this call will also update pair cache with recent examples
        
        #convert to numpy array
        x = np.array(example_window, dtype=np.float32)
        x[:, 0] = timestamp_to_percent(x[:, 0])

        # standardize using same mean/std from creation of model
        x = standard_scalar.transform(x)

        x = vectorize_window(x, window_length)

        # Reshape the example to add a batch dimension
        x = np.reshape(x, (1,) + x.shape)

        prediction = model(x)

        print(prediction)
    
# Function to handle incoming messages
def on_message(ws, message):
    global examples_received
    global raw_ticker_stream
    # Parse the incoming message (assuming it's JSON)
    result = json.loads(message)
    # Process the data as needed
    data = result[1]
    pair = result[3]
    now = str(datetime.datetime.now())
    raw_ticker_stream.append({now: {'pair': pair, 'data': data}})
    examples_received = examples_received + 1

# Function to handle WebSocket open event
def on_open(ws):
    print("WebSocket connection opened")
    'XBT/USD'
    # Subscribe to the desired channel(s) after the connection is open
    # Send the subscription message to the WebSocket server
    subscription_message = {
        "event": "subscribe",
        "pair": get_all_wsnames(),
        "subscription": {
            "name": "ticker"
        }
    }
    ws.send(json.dumps(subscription_message))

# Function to handle WebSocket close event
def on_close(ws):
    print("WebSocket connection closed")
    # join threads
    model_thread.join()

"""
Need separate threads of execution here which, upon reaching NUM_EXAMPLES messages, will create a model from these examples.
As the model is being created, more data will be streaming in. Call this extra data X_extra. 
After the model is created, the main thread should go to the most recent X_extra examples and start prediction. 

Question: Should there be "backfilling" of this unused X_extra data? Presumably this thread could online-train 
this extra data in between attempts at predicting the most recent data, considering that we are throttled by 
120 trades per minute. On first cut this data will just be lost. 

THe trading thread should also be checking the mse score for incoming data, and if the score deteriorates past a certain threshold,
should tell the model thread to train a new model with more recent data. This implies that data that we run predictions on should
be stored for future model training. 
"""

model_thread = threading.Thread(target=model_thread_func)

model_thread.start()

# Create a WebSocket connection
ws = websocket.WebSocketApp("wss://ws.kraken.com/", on_message=on_message, on_open=on_open, on_close=on_close)

# Start the WebSocket connection (runs in a separate thread)
ws.run_forever() # this hangs until websocket is stopped

# Close websocket gracefully
ws.close()