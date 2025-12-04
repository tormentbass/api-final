# api_client.py
import os
import requests
from typing import Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.environ.get("RAPIDAPI_HOST", "api-football-v1.p.rapidapi.com")
BASE_URL = os.environ.get("BASE_URL", "https://api-football-v1.p.rapidapi.com/v3/")


HEADERS = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": RAPIDAPI_HOST
}

# Session com retry
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

def buscar_dados_partida(match_id: int) -> Optional[dict]:
    if not RAPIDAPI_KEY:
        raise RuntimeError("RAPIDAPI_KEY não configurada no ambiente.")
    endpoint = f"{BASE_URL}fixtures"
    params = {"id": match_id}
    try:
        resp = session.get(endpoint, headers=HEADERS, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # estrutura: data['response'] -> list
        if data.get("results", 0) > 0 and data.get("response"):
            return data["response"][0]
        return None
    except requests.HTTPError as e:
        print(f"[api_client] HTTPError: {e} - status {getattr(e.response,'status_code',None)}")
        return None
    except Exception as e:
        print(f"[api_client] Erro desconhecido: {e}")
        return None
