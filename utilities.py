import requests

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