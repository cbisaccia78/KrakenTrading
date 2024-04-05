import requests

def get_currency_pairs():
    url = "https://api.kraken.com/0/public/AssetPairs"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()

        if 'error' in data:
            print("API Error:", data['error'])
            return None

        currency_pairs = list(data['result'].keys())
        return currency_pairs

    except requests.exceptions.RequestException as e:
        print("Request Error:", e)
        return None

resp = requests.get('https://api.kraken.com/0/public/Assets')

print(resp.json())
