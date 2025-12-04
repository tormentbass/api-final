# api_client.py
import os
import requests
from typing import Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ===============================
# VARIÁVEIS DE AMBIENTE
# ===============================
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.environ.get("RAPIDAPI_HOST", "api-football-v1.p.rapidapi.com")
BASE_URL = os.environ.get("BASE_URL", "https://api-football-v1.p.rapidapi.com/v3/")

# ===============================
# DEBUG – ISSO VAI APARECER NOS LOGS DO RENDER
# ===============================
print("====== DEBUG VARIÁVEIS DE AMBIENTE ======")
print("RAPIDAPI_KEY está definida? ", "SIM" if RAPIDAPI_KEY else "NÃO!")
print("RAPIDAPI_HOST:", RAPIDAPI_HOST)
print("BASE_URL:", BASE_URL)
print("=========================================")

HEADERS = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": RAPIDAPI_HOST
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
# FUNÇÃO PRINCIPAL: BUSCAR DADOS REAIS DA PARTIDA
# ===================================================
def buscar_dados_partida(match_id: int) -> Optional[dict]:
    # debug inicial
    print(f"\n[DEBUG] Buscando dados da partida ID = {match_id}")
    print("[DEBUG] Endpoint montado:", f"{BASE_URL}fixtures")
    print("[DEBUG] HEADERS enviados:", HEADERS)

    if not RAPIDAPI_KEY:
        raise RuntimeError("RAPIDAPI_KEY não configurada no ambiente!")

    endpoint = f"{BASE_URL}fixtures"
    params = {"id": match_id}

    try:
        resp = session.get(endpoint, headers=HEADERS, params=params, timeout=10)
        print("[DEBUG] Status code recebido:", resp.status_code)

        resp.raise_for_status()
        data = resp.json()

        # debug dados crus
        print("[DEBUG] JSON recebido:", data)

        if data.get("results", 0) > 0 and data.get("response"):
            return data["response"][0]

        print("[DEBUG] Nenhum dado encontrado para essa partida.")
        return None

    except requests.HTTPError as e:
        print(f"[api_client] HTTPError: {e} - status {getattr(e.response,'status_code',None)}")
        return None

    except Exception as e:
        print(f"[api_client] Erro desconhecido: {e}")
        return None
