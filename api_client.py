# ====================================================================================================
# ARQUIVO: api_client.py
# VERSÃO: 1.0.0
# CRIADO: 06/12/2025
# DESC: Cliente API-Football robusto (retries, validação, logs)
# ====================================================================================================

import os
import requests
from typing import Optional, Dict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# CONFIG via ENV
API_KEY = os.getenv("API_FOOTBALL_KEY")           # chave x-apisports-key
API_HOST = os.getenv("API_FOOTBALL_HOST", "v3.football.api-sports.io")
BASE_URL = os.getenv("API_FOOTBALL_BASE", "https://v3.football.api-sports.io")

# DEBUG (aparecerá nos logs do Render)
print("====== DEBUG API-FOOTBALL OFFICIAL ======")
print("API KEY definida?", "SIM" if API_KEY else "NÃO!")
print("HOST:", API_HOST)
print("BASE_URL:", BASE_URL)
print("=========================================")

HEADERS = {
    "x-apisports-key": API_KEY,
    # "x-rapidapi-host": API_HOST   # opcional; algumas contas ainda usam rapidapi header
}

# Sessão com retry
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.6, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retries)
session.mount("https://", adapter)
session.mount("http://", adapter)


def _safe_get(url: str, params: dict = None, timeout: int = 10) -> Optional[Dict]:
    """Helper: faz GET com session, retorna JSON ou None e loga erros."""
    try:
        resp = session.get(url, headers=HEADERS, params=params, timeout=timeout)
        # status log (aparece no Render)
        print(f"[api_client] GET {url} params={params} -> status {resp.status_code}")
        text = resp.text[:2000] if resp.text else ""
        print(f"[api_client] response_preview: {text}")
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as e:
        # log detalhado
        status = getattr(e.response, "status_code", None)
        print(f"[api_client] HTTPError ({status}): {e} - content: {getattr(e.response,'text',None)}")
        return None
    except Exception as e:
        print(f"[api_client] Erro geral ao chamar API-Football: {e}")
        return None


def buscar_dados_partida(match_id: int) -> Optional[dict]:
    """
    Busca dados completos da partida (fixture) pelo fixture id (match_id).
    Retorna dicionário com dados originais (response[0]) ou None.
    """
    if not API_KEY:
        raise RuntimeError("API_FOOTBALL_KEY não configurada no ambiente!")

    endpoint = f"{BASE_URL}/fixtures"
    params = {"id": match_id}

    data = _safe_get(endpoint, params=params, timeout=12)
    if data is None:
        return None

    # A API retorna {"response": [ {...} ], "results": N}
    response_list = data.get("response") or []
    if len(response_list) == 0:
        print(f"[api_client] Nenhuma partida encontrada para id={match_id}")
        return None

    # retorno principal
    fixture = response_list[0]

    # Sanity checks mínimos (garantir campos essenciais)
    # fixture pode ter estrutura: fixture, league, teams, goals, score, events, lineups, statistics, players
    essential = ["fixture", "league", "teams", "score"]
    for e in essential:
        if e not in fixture:
            print(f"[api_client] Atenção: campo essencial '{e}' ausente no fixture id={match_id}")

    return fixture


# função utilitária (opcional) para buscar fixtures por dia / liga, se precisar
def buscar_fixtures_por_data(date_str: str, league_id: Optional[int] = None, status: Optional[str] = None) -> Optional[dict]:
    """
    Ex: date_str = "2025-12-06"
    league_id opcional, status opcional (e.g. "NS", "1H", "FT", "LIVE")
    """
    if not API_KEY:
        raise RuntimeError("API_FOOTBALL_KEY não configurada no ambiente!")

    endpoint = f"{BASE_URL}/fixtures"
    params = {"date": date_str}
    if league_id:
        params["league"] = league_id
    if status:
        params["status"] = status

    return _safe_get(endpoint, params=params, timeout=12)
