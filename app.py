# ====================================================================================================
# ARQUIVO: app.py - VERSÃO FINAL CORRIGIDA
# ====================================================================================================

import json
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from api_client import buscar_dados_partida, buscar_fixtures_por_data
from engine import gerar_relatorio_json, gerar_ranking_forca

# 1. INICIALIZAÇÃO DO APP
app = FastAPI()

# 2. CONFIGURAÇÃO DE CORS (ESSENCIAL PARA O LOVABLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================================================================================================
# CARREGA MAPEAMENTO DE NOMES
# ====================================================================================================
NAME_MAP_PATH = "name_map.json"
if os.path.exists(NAME_MAP_PATH):
    with open(NAME_MAP_PATH, "r", encoding="utf-8") as f:
        NAME_MAP = json.load(f)
else:
    NAME_MAP = {}

def normalizar_nome(nome: str) -> str:
    if not nome: return nome
    nome = nome.strip()
    if nome in NAME_MAP: return NAME_MAP[nome]
    for k, v in NAME_MAP.items():
        if k.lower() == nome.lower(): return v
    return nome

# ====================================================================================================
# MODELOS DE DADOS (PYDANTIC)
# ====================================================================================================
class LiveMatchRequest(BaseModel):
    match_id: int

class PredictTeamsRequest(BaseModel):
    time_casa: str
    time_fora: str
    data_base: str

# ====================================================================================================
# ENDPOINTS
# ====================================================================================================

@app.get("/")
async def root():
    return {"status": "SaaS de Analise Online", "engine": "Rodando"}

@app.get("/jogos-hoje")
async def listar_jogos_do_dia():
    from datetime import date
    hoje = date.today().isoformat()
    try:
        dados = buscar_fixtures_por_data(hoje)
        if not dados or "response" not in dados:
            return {"status": "vazio", "jogos": []}
        return {"status": "sucesso", "data": hoje, "jogos": dados["response"]}
    except Exception as e:
        return {"status": "erro", "mensagem": str(e)}

@app.post("/predict/live")
async def predict_live(request: LiveMatchRequest):
    fixture = buscar_dados_partida(request.match_id)
    if fixture is None:
        return {"erro": "Não foi possível obter fixture para esse match_id."}

    home = fixture["teams"]["home"]["name"]
    away = fixture["teams"]["away"]["name"]
    home_norm = normalizar_nome(home)
    away_norm = normalizar_nome(away)
    data_base = fixture["fixture"]["date"].split("T")[0]

    resultado = gerar_relatorio_json(home_norm, away_norm, data_base)
    return {
        "originais": {"home": home, "away": away},
        "normalizados": {"home": home_norm, "away": away_norm},
        "analise": resultado
    }

@app.get("/stats/rankings")
async def rankings(data_base: str = "2025-01-01"):
    return gerar_ranking_forca(data_base)

@app.get("/test-api-football")
async def test_api():
    result = buscar_dados_partida(710691)
    return {"resultado": "ok" if result else "falhou", "response": result}

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
