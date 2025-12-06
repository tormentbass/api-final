# ====================================================================================================
# ARQUIVO: app.py
# VERSÃO: 2.0.0
# DATA: 06/12/2025 - FULL INTEGRATION
# DESCRIÇÃO:
#     - Integra API-Football → engine ML
#     - Mapeamento automático de nomes
#     - Endpoints live, batch, debug, rankings
# ====================================================================================================

import json
import os
from fastapi import FastAPI
from pydantic import BaseModel

from api_client import buscar_dados_partida
from engine import gerar_relatorio_json, gerar_ranking_forca

# ====================================================================================================
# CARREGA MAPEAMENTO DE NOMES
# ====================================================================================================
NAME_MAP_PATH = "name_map.json"

if os.path.exists(NAME_MAP_PATH):
    with open(NAME_MAP_PATH, "r", encoding="utf-8") as f:
        NAME_MAP = json.load(f)
        print(f"[APP] Name map carregado: {len(NAME_MAP)} entradas")
else:
    NAME_MAP = {}
    print("[APP] AVISO: name_map.json não encontrado, usando mapa vazio.")


def normalizar_nome(nome: str) -> str:
    """Tenta corrigir nomes de times usando name_map."""
    if not nome:
        return nome

    nome = nome.strip()

    # 1) Match direto
    if nome in NAME_MAP:
        return NAME_MAP[nome]

    # 2) Match case-insensitive
    for k, v in NAME_MAP.items():
        if k.lower() == nome.lower():
            return v

    # 3) Normalização simples
    nome_norm = nome.replace(" FC", "").replace("CF", "").replace(".", "").strip()
    if nome_norm in NAME_MAP:
        return NAME_MAP[nome_norm]

    return nome  # fallback


# ====================================================================================================
# FASTAPI
# ====================================================================================================
app = FastAPI()


# ====================================================================================================
# MODELOS
# ====================================================================================================
class LiveMatchRequest(BaseModel):
    match_id: int


class PredictTeamsRequest(BaseModel):
    time_casa: str
    time_fora: str
    data_base: str


# ====================================================================================================
# ENDPOINT: Testar conexão com API-Football
# ====================================================================================================
@app.get("/test-api-football")
async def test_api():
    result = buscar_dados_partida(710691)  # fixture conhecido
    return {"resultado": "ok" if result else "falhou", "response": result}


# ====================================================================================================
# ENDPOINT: Retorna fixture cru da API-Football
# ====================================================================================================
@app.get("/partida/{match_id}")
async def get_partida(match_id: int):
    fixture = buscar_dados_partida(match_id)
    if fixture is None:
        return {"erro": "Partida não encontrada ou erro na API-Football."}
    return fixture


# ====================================================================================================
# ENDPOINT: Previsão Live com API-Football
# ====================================================================================================
@app.post("/predict/live")
async def predict_live(request: LiveMatchRequest):

    fixture = buscar_dados_partida(request.match_id)
    if fixture is None:
        return {"erro": "Não foi possível obter fixture para esse match_id."}

    home = fixture["teams"]["home"]["name"]
    away = fixture["teams"]["away"]["name"]

    # Normalização e map
    home_norm = normalizar_nome(home)
    away_norm = normalizar_nome(away)

    print(f"[LIVE] home={home} -> {home_norm}")
    print(f"[LIVE] away={away} -> {away_norm}")

    # Usa data do fixture
    data_base = fixture["fixture"]["date"].split("T")[0]

    resultado = gerar_relatorio_json(home_norm, away_norm, data_base)
    return {
        "originais": {"home": home, "away": away},
        "normalizados": {"home": home_norm, "away": away_norm},
        "analise": resultado
    }


# ====================================================================================================
# ENDPOINT: Previsão manual (nomes + data)
# ====================================================================================================
@app.post("/predict/teams")
async def predict_teams(request: PredictTeamsRequest):

    casa = normalizar_nome(request.time_casa)
    fora = normalizar_nome(request.time_fora)

    return {
        "entrada": {
            "original": {"casa": request.time_casa, "fora": request.time_fora},
            "normalizado": {"casa": casa, "fora": fora},
        },
        "analise": gerar_relatorio_json(casa, fora, request.data_base),
    }


# ====================================================================================================
# ENDPOINT: Diagnóstico de nomes
# ====================================================================================================
@app.get("/debug/nome/{nome}")
async def debug_nome(nome: str):
    nome_norm = normalizar_nome(nome)

    return {
        "entrada": nome,
        "normalizado": nome_norm,
        "map_direct": NAME_MAP.get(nome),
        "map_ci_match": [k for k in NAME_MAP if k.lower() == nome.lower()],
    }


# ====================================================================================================
# ENDPOINT: Rankings
# ====================================================================================================
@app.get("/stats/rankings")
async def rankings(data_base: str = "2025-01-01"):
    return gerar_ranking_forca(data_base)
