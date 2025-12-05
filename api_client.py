import os
import requests
from typing import Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ===============================
# VARIÁVEIS DE AMBIENTE (NOVO)
# ===============================
API_KEY = os.getenv("API_FOOTBALL_KEY")
API_HOST = os.getenv("API_FOOTBALL_HOST", "v3.football.api-sports.io")
BASE_URL = os.getenv("API_FOOTBALL_BASE", "https://v3.football.api-sports.io")

print("====== DEBUG API-FOOTBALL OFFICIAL ======")
print("API KEY definida?", "SIM" if API_KEY else "NÃO!")
print("HOST:", API_HOST)
print("BASE_URL:", BASE_URL)
print("=========================================")

HEADERS = {
    "x-apisports-key": API_KEY,
    "x-rapidapi-host": API_HOST  # API Sports ainda usa esse header, mas próprio deles
}

# Sessão com retry inteligente
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
    print(f"[DEBUG] Buscando partida (API-Football Oficial): ID = {match_id}")

    if not API_KEY:
        raise RuntimeError("API_FOOTBALL_KEY não configurada!")

    endpoint = f"{BASE_URL}/fixtures"
    params = {"id": match_id}

    print("[DEBUG] Endpoint:", endpoint)
    print("[DEBUG] Params:", params)
    print("[DEBUG] Headers:", HEADERS)

    try:
        resp = session.get(endpoint, headers=HEADERS, params=params, timeout=10)
        print("[DEBUG] Status:", resp.status_code)
        print("[DEBUG] Response raw:", resp.text)

        resp.raise_for_status()
        data = resp.json()

        if data.get("response"):
            return data["response"][0]

        print("[DEBUG] Nenhum dado encontrado.")
        return None

    except Exception as e:
        print("[DEBUG] ERRO API CLIENTE:", e)
        return None
