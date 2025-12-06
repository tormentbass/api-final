# ====================================================================================================
# ARQUIVO: app.py
# VERSÃO: 1.0.0
# CRIADO EM: 06/12/2025 - 11:32h
# AUTOR: Plínio (com engenharia ChatGPT)
#
# DESCRIÇÃO:
#     API principal do projeto "Análise Esportiva IA".
#     - Serve como gateway entre o modelo interno (engine.py)
#     - Busca dados reais usando API-Football (api_client.py)
#     - Preparada para integração futura com Supabase e Lovable
#
# OBS IMPORTANTE:
#     Este arquivo segue padrões de produção compatíveis com Render, Vercel, Railway e Lovable.
# ====================================================================================================

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import os
import requests

import engine
from api_client import buscar_dados_partida


# ====================================================================================================
# INICIALIZAÇÃO DA API
# ====================================================================================================

app = FastAPI(
    title="Análise Esportiva IA",
    description="API de Previsões e Rankings - Modelo + Dados Reais",
    version="1.0.0"
)


# ====================================================================================================
# 1) ENDPOINT DE SAÚDE
# ====================================================================================================
@app.get("/", tags=["Sistema"])
def root():
    status = "OK" if engine.model is not None else "ERRO"
    return {"api_status": status, "message": "Motor de Análise Esportiva ativo!"}


# ====================================================================================================
# 2) SISTEMA 1 – PREVISÃO VIA IA INTERNA
# ====================================================================================================

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


# ====================================================================================================
# 3) SISTEMA 2 – PREVISÃO COM DADOS REAIS (API-FOOTBALL)
# ====================================================================================================

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


# ====================================================================================================
# 4) CONSULTA SIMPLES DE PARTIDA
# ====================================================================================================
@app.get("/partida/{match_id}", tags=["Dados - API Externa"])
async def partida(match_id: int):
    dados = buscar_dados_partida(match_id)

    if not dados:
        raise HTTPException(status_code=404, detail="Partida não encontrada")

    return dados


# ====================================================================================================
# 5) RANKINGS – GERADO PELO MOTOR INTERNO
# ====================================================================================================
@app.get("/stats/rankings", tags=["Rankings"])
async def rankings(date: str = Query(...)):
    try:
        ranks = engine.gerar_ranking_forca(date)
        return ranks

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ====================================================================================================
# 6) TESTE COMPLETO DA API-FOOTBALL
# ====================================================================================================
@app.get("/test-api-football", tags=["Debug"])
def test_api_football():
    try:
        key = os.getenv("API_FOOTBALL_KEY")
        host = os.getenv("API_FOOTBALL_HOST", "v3.football.api-sports.io")

        url = "https://v3.football.api-sports.io/status"

        headers = {
            "x-apisports-key": key,
            "x-rapidapi-host": host
        }

        r = requests.get(url, headers=headers, timeout=10)

        return {
            "status_code": r.status_code,
            "response": r.text,
            "key_last4": key[-4:] if key else "NONE",
            "host_used": host
        }

    except Exception as e:
        return {"error": str(e)}
