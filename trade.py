import websocket
import json
import datetime
import threading

import numpy as np

from api import get_all_wsnames
from high_bid_model import create_model, FEATURE_MAP
from utilities import vectorize_from_cache, timestamp_to_percent, vectorize_windows

PING = {
    "event": "ping",
    "reqid": 69420
}

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
errors = []

count = 0

NUM_EXAMPLES = 4459

def model_thread_func():
    # TODO - need to keep the pair cache up to data even if we aren't updating model

    global model
    global raw_ticker_stream
    global count
    global retrain
    global errors

    if model is None:
            if count >= NUM_EXAMPLES:
                model, test_mse, standard_scalar, pair_cache = create_model(raw_ticker_stream, pair_name, window_len=window_length)
    elif retrain:
        count = 0
        model = None
    else: # we can trade
        # grab the most recent window_length examples
        recent_examples = raw_ticker_stream[count-5:]
        recent_tickers = vectorize_from_cache(pair_cache, recent_examples)
        
        #convert to numpy array
        x = np.array(recent_tickers, dtype=np.float32)
        x[:, 0] = timestamp_to_percent(x[:, 0])

        # standardize using same mean/std from creation of model
        x = standard_scalar.transform(x)

        x = vectorize_windows(x, window_length)

        # Reshape the example to add a batch dimension
        x = np.reshape(x, (1,) + x.shape)

        prediction = model(x)

        print(prediction)
    
# Function to handle incoming messages
def on_message(ws, message):
    global count
    global raw_ticker_stream
    # Parse the incoming message (assuming it's JSON)
    result = json.loads(message)
    # Process the data as needed
    data = result[1]
    pair = result[3]
    now = str(datetime.datetime.now())
    raw_ticker_stream.append({now: {'pair': pair, 'data': data}})
    count = count + 1

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