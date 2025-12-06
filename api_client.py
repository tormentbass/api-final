# ============================================
# Versão: 1.0.0
# Criado em: 06/12/2025 - 10:45h
# Autor: Plínio
# Descrição: API Client otimizado pro Render + Supabase + Lovable
# ============================================

import os
import requests
from typing import Optional, Dict, Any
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============================================
# CONFIGURAÇÕES BÁSICAS VIA VARIÁVEIS DE AMBIENTE
# ============================================

API_KEY = os.getenv("API_FOOTBALL_KEY")
API_HOST = os.getenv("API_FOOTBALL_HOST", "v3.football.api-sports.io")
BASE_URL = os.getenv("API_FOOTBALL_BASE", "https://v3.football.api-sports.io")

ENV = os.getenv("ENVIRONMENT", "development")  # development / production

if ENV == "development":
    print("\n========== API CLIENT DEBUG ==========")
    print("API KEY definida? ", "SIM" if API_KEY else "NÃO!!")
    print("HOST:", API_HOST)
    print("BASE_URL:", BASE_URL)
    print("======================================\n")

# ============================================
# HEADERS GLOBAIS
# ============================================

HEADERS = {
    "x-apisports-key": API_KEY,
    "x-rapidapi-host": API_HOST,
    "Accept": "application/json"
}

# ============================================
# CRIAÇÃO DE SESSÃO GLOBAL COM RETRY + TIMEOUT
# ============================================

def create_session() -> requests.Session:
    session = requests.Session()

    retries = Retry(
        total=5,
        backoff_factor=0.4,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )

    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session

session = create_session()

# ============================================
# FUNÇÃO UNIVERSAL DE CHAMADA À API
# (Serve agora e serve para futuras APIs também)
# ============================================

def call_api(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict]:
    url = f"{BASE_URL}/{endpoint}"

    if ENV == "development":
        print(f"[API][DEBUG] URL → {url}")
        print(f"[API][DEBUG] Params → {params}")
        print(f"[API][DEBUG] Headers → {HEADERS}")

    try:
        response = session.get(
            url,
            headers=HEADERS,
            params=params or {},
            timeout=12
        )

        if ENV == "development":
            print(f"[API][DEBUG] Status code → {response.status_code}")
            print(f"[API][DEBUG] Raw response → {response.text[:500]}")

        response.raise_for_status()
        data = response.json()

        # Para API Football, tudo vem no campo "response"
        return data

    except Exception as e:
        print("[API][ERRO] Falha ao chamar API:", e)
        return None

# ============================================
# FUNÇÃO PRINCIPAL — BUSCAR PARTIDA
# ============================================

def buscar_dados_partida(match_id: int) -> Optional[Dict]:
    """
    Retorna os dados brutos de uma partida.
    O Lovable vai consumir isso depois.
    """
    if not API_KEY:
        raise RuntimeError("API_FOOTBALL_KEY não configurada!")

    data = call_api("fixtures", params={"id": match_id})

    if not data:
        return None

    if data.get("response"):
        return data["response"][0]

    return None


# ============================================
# EXTENSÍVEL PARA O FUTURO
# — Buscar jogadores
# — Buscar times
# — Buscar estatísticas
# — Buscar campeonatos
# — Buscar cotejos
# ============================================

def buscar_time(team_id: int) -> Optional[Dict]:
    return call_api("teams", params={"id": team_id})


def buscar_jogadores(team_id: int) -> Optional[Dict]:
    return call_api("players", params={"team": team_id})


def buscar_estatisticas(match_id: int) -> Optional[Dict]:
    return call_api("fixtures/statistics", params={"fixture": match_id})
