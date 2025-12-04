# ==============================================================================
# ARQUIVO: app.py
# ==============================================================================
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List

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
# 2) SISTEMA 1: IA TRADICIONAL (Chelsea x Arsenal)
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
# 3) SISTEMA 2: DADOS REAIS (API-Football)
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
# 4) ROTA DE TESTE (GET) → usado pelo Lovable
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


#teste da API

@app.route("/test-api-football")
def test_api_football():
    import requests, os

    url = "https://api-football-v1.p.rapidapi.com/v3/timezone"
    headers = {
        "x-rapidapi-key": os.getenv("API_FOOTBALL_KEY"),
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
    }

    try:
        r = requests.get(url, headers=headers, timeout=10)
        return {
            "status": r.status_code,
            "response": r.text,
            "key_empty": os.getenv("API_FOOTBALL_KEY") is None
        }
    except Exception as e:
        return {"error": str(e)}

