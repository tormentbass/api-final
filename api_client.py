# api_client.py
import os
import requests
from typing import Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ===============================
# VARIÁVEIS DE AMBIENTE (CORRETAS)
# ===============================
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")  # CHAVE CERTA
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST", "api-football-v1.p.rapidapi.com")

BASE_URL = "https://api-football-v1.p.rapidapi.com/v3/"


# ===============================
# DEBUG NO RENDER
# ===============================
print("====== DEBUG ENV ======")
print("API_FOOTBALL_KEY definida?", "SIM" if API_FOOTBALL_KEY else "NÃO!")
print("RAPIDAPI_HOST:", RAPIDAPI_HOST)
print("BASE_URL:", BASE_URL)
print("========================")


HEADERS = {
    "x-rapidapi-key": API_FOOTBALL_KEY,
    "x-rapidapi-host": RAPIDAPI_HOST
}

# Sessão com retry
session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504]
)
session.mount("https://", HTTPAdapter(max_retries=retries))


# ===================================================
# FUNÇÃO PRINCIPAL: BUSCAR DADOS DA PARTIDA
# ===================================================
def buscar_dados_partida(match_id: int) -> Optional[dict]:

    print(f"\n[API] Buscando partida {match_id}")
    print("[API] HEADERS:", HEADERS)

    if not API_FOOTBALL_KEY:
        raise RuntimeError("API_FOOTBALL_KEY não configurada no Render!")

    endpoint = f"{BASE_URL}fixtures"
    params = {"id": match_id}

    try:
        resp = session.get(endpoint, headers=HEADERS, params=params, timeout=10)
        print("[API] Status:", resp.status_code)

        resp.raise_for_status()
        data = resp.json()

        if (
            data.get("results", 0) > 0
            and isinstance(data.get("response"), list)
            and len(data["response"]) > 0
        ):
            return data["response"][0]

        print("[API] Nenhum dado encontrado para essa partida.")
        return None

    except requests.HTTPError as e:
        print(f"[API] HTTPError: {e}")
        return None

    except Exception as e:
        print(f"[API] Erro desconhecido:", e)
        return None
