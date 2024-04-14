import websocket
import json
import time

PING = {
    "event": "ping",
    "reqid": 69420
}

PUBLIC_URL = "wss://ws.kraken.com/"
PRIVATE_URL = "wss://ws-auth.kraken.com/"


# Function to handle incoming messages
def on_message(ws, message):
    # Parse the incoming message (assuming it's JSON)
    data = json.loads(message)
    # Process the data as needed
    print("Received message:", data)

# Function to handle WebSocket open event
def on_open(ws):
    print("WebSocket connection opened")
    # Subscribe to the desired channel(s) after the connection is open
    # Send the subscription message to the WebSocket server
    subscription_message = {
        "event": "subscribe",
        "pair": ["XBT/USD"],  # Example: Subscribe to Bitcoin/USD pair
        "subscription": {
            "name": "trade"
        }
    }
    ws.send(json.dumps(subscription_message))

# Function to handle WebSocket close event
def on_close(ws):
    print("WebSocket connection closed")

""" # Function to handle WebSocket error event
def on_error(ws, error):
    print("WebSocket error:", error)
    # Attempt to reconnect after a short delay
    print("Attempting to reconnect in 5 seconds...")
    time.sleep(5)
    ws.run_forever() """

# Create a WebSocket connection
ws = websocket.WebSocketApp("wss://ws.kraken.com/", on_message=on_message, on_open=on_open, on_close=on_close)

# Start the WebSocket connection (runs in a separate thread)
ws.run_forever()

# Close websocket gracefully
ws.close()