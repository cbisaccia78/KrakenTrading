import requests
import hashlib
import hmac
import base64
import time
import urllib

url = 'https://api.kraken.com'
api_key = 'your_key_here'
api_secret = 'your_secret_here'


def get_kraken_signature(urlpath, data, secret):

    postdata = urllib.parse.urlencode(data)
    encoded = (str(data['nonce']) + postdata).encode()
    message = urlpath.encode() + hashlib.sha256(encoded).digest()

    mac = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
    sigdigest = base64.b64encode(mac.digest())
    return sigdigest.decode()


# Attaches auth headers and returns results of a POST request
def _kraken_request(uri_path, data, api_key, api_sec):
    headers = {}
    headers['API-Key'] = api_key
    # get_kraken_signature() as defined in the 'Authentication' section
    headers['API-Sign'] = get_kraken_signature(uri_path, data, api_sec)
    req = requests.post((url + uri_path), headers=headers, data=data)
    return req

def kraken_request(uri_path, data):
    return _kraken_request(uri_path, data, api_key, api_secret)


def get_ws_auth_token():
    """
        returns:
            {'error': [], 'result': {'expires': time in seconds, 'token': token}}
    """
    resp = kraken_request('/0/private/GetWebSocketsToken', {
        "nonce": str(int(1000*time.time()))
    }, api_key, api_secret)

    return resp