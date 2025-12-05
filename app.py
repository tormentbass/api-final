# ==============================================================================
# ARQUIVO: app.py
# ==============================================================================
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import os
import requests

import engine
from api_client import buscar_dados_partida


app = FastAPI(
    title="Análise Esportiva IA",
    description="API de Previsões e Rankings",
    version="1.0.0"
)

# ------------------------------------------------------------------------------
# 1) ENDPOINT DE SAÚDE
# ------------------------------------------------------------------------------
@app.get("/")
def root():
    status = "OK" if engine.model is not None else "ERRO"
    return {"api_status": status, "message": "Motor de Análise Esportiva ativo!"}


# ------------------------------------------------------------------------------
# 2) SISTEMA 1 – PREVISÃO IA TRADICIONAL
# ------------------------------------------------------------------------------
class MatchInputIA(BaseModel):
    home_team: str
    away_team: str
    date: str


@app.post("/predict/teams", tags=["Previsão - IA Interna"])
async def predict_teams(match: MatchInputIA):
    try:
        relatorio = engine.gerar_relatorio_json(
            match.home_team,
            match.away_team,
            match.date
        )
        return relatorio
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------------------
# 3) SISTEMA 2 – DADOS VIA API-FOOTBALL
# ------------------------------------------------------------------------------
class MatchInputID(BaseModel):
    match_id: int


@app.post("/predict/match", tags=["Previsão - Dados Reais"])
async def predict_match(match: MatchInputID):
    try:
        dados = buscar_dados_partida(match.match_id)
        if not dados:
            raise HTTPException(status_code=404, detail="Partida não encontrada")
        return {"status": "SUCESSO", "dados": dados}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------------------
# 4) CONSULTA DE PARTIDA – TESTE SIMPLES
# ------------------------------------------------------------------------------
@app.get("/partida/{match_id}", tags=["Dados - API Externa"])
async def partida(match_id: int):
    dados = buscar_dados_partida(match_id)
    if not dados:
        raise HTTPException(status_code=404, detail="Partida não encontrada")
    return dados


# ------------------------------------------------------------------------------
# 5) RANKINGS
# ------------------------------------------------------------------------------
@app.get("/stats/rankings", tags=["Rankings"])
async def rankings(date: str = Query(...)):
    try:
        ranks = engine.gerar_ranking_forca(date)
        return ranks
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------------------
# 6) TESTE COMPLETO DA API-FOOTBALL (debug real)
# ------------------------------------------------------------------------------
@app.get("/test-api-football", tags=["Debug"])
def test_api_football():

    API_KEY = os.getenv("API_FOOTBALL_KEY")
    API_HOST = os.getenv("RAPIDAPI_HOST", "api-football-v1.p.rapidapi.com")

    if not API_KEY:
        return {
            "error": "API_FOOTBALL_KEY not found",
            "fix": "Defina a variável no Render → Environment → API_FOOTBALL_KEY"
        }

    url = "https://api-football-v1.p.rapidapi.com/v3/status"

    headers = {
        "x-rapidapi-key": API_KEY,
        "x-rapidapi-host": API_HOST
    }

    try:
        r = requests.get(url, headers=headers, timeout=10)

        return {
            "status_code": r.status_code,
            "response": r.text,
            "key_last4": API_KEY[-4:], 
            "host_used": API_HOST
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
