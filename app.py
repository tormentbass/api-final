# ====================================================================================================
# ARQUIVO: app.py
# VERSÃO: 1.1.0
# CRIADO: 06/12/2025
# AUTOR: Plínio + ChatGPT
#
# ENDPOINTS:
# - GET  /                     -> health
# - GET  /test-api-football    -> debug api-football
# - GET  /partida/{match_id}   -> dados crus do fixture (já existia)
# - POST /predict/match        -> previsão usando input manual (já existia)
# - POST /predict/match-full   -> NOVO: pega dados reais, trata, roda modelo e retorna relatório
# ====================================================================================================

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
import os
import requests
import datetime

import engine
from api_client import buscar_dados_partida

app = FastAPI(
    title="Análise Esportiva IA",
    description="API de Previsões e Rankings - Modelo + Dados Reais",
    version="1.1.0"
)


# ----------------------------
# Models
# ----------------------------
class MatchInputIA(BaseModel):
    home_team: str
    away_team: str
    date: str


class MatchInputID(BaseModel):
    match_id: int
    include_raw: Optional[bool] = False   # se true, retornamos também os dados brutos da API-Football


# ----------------------------
# Root / Health
# ----------------------------
@app.get("/", tags=["Sistema"])
def root():
    status = "OK" if engine.model is not None else "ERRO"
    return {"api_status": status, "message": "Motor de Análise Esportiva ativo!"}


# ----------------------------
# Predict teams (IA interna) - já existente
# ----------------------------
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


# ----------------------------
# Predict match (dados reais) - já existente
# ----------------------------
@app.post("/predict/match", tags=["Previsão - Dados Reais"])
async def predict_match(match: MatchInputID):
    try:
        dados = buscar_dados_partida(match.match_id)
        if not dados:
            raise HTTPException(status_code=404, detail="Partida não encontrada")
        return {"status": "SUCESSO", "dados": dados}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# NOVO: /predict/match-full
# Faz a pipeline completa: busca dados -> normaliza -> executa engine -> retorna relatório padronizado
# ----------------------------
@app.post("/predict/match-full", tags=["Previsão - Pipeline Completa"])
async def predict_match_full(payload: MatchInputID):
    """
    Fluxo:
    1) buscar_dados_partida(match_id)
    2) extrair home_team, away_team, date (fallbacks se faltar)
    3) validar e chamar engine.gerar_relatorio_json(home, away, date)
    4) montar resposta padronizada (meta + previsao + raw opcional)
    """
    match_id = payload.match_id
    include_raw = bool(payload.include_raw)

    try:
        # 1) buscar dados crus
        raw = buscar_dados_partida(match_id)
        if not raw:
            raise HTTPException(status_code=404, detail=f"Partida {match_id} não encontrada na API-Football")

        # 2) extrair campos essenciais com fallback
        # fixture.date é padrão: "2021-12-01T19:30:00+00:00"
        fixture = raw.get("fixture", {})
        teams = raw.get("teams", {})
        league = raw.get("league", {})

        # extrair nomes
        home_team_name = None
        away_team_name = None
        try:
            home_team_name = teams.get("home", {}).get("name")
            away_team_name = teams.get("away", {}).get("name")
        except Exception:
            pass

        # extrair data
        date_raw = fixture.get("date") or fixture.get("timestamp")
        # prefer string ISO; se for timestamp, converte
        if isinstance(date_raw, int):
            # timestamp -> ISO
            date_iso = datetime.datetime.utcfromtimestamp(date_raw).isoformat()
        else:
            date_iso = date_raw

        # validações básicas
        if not home_team_name or not away_team_name or not date_iso:
            # ainda assim podemos tentar extrair de outros campos (fallback)
            # mas se faltar algo crítico, devolvemos erro específico
            missing = []
            if not home_team_name:
                missing.append("home_team")
            if not away_team_name:
                missing.append("away_team")
            if not date_iso:
                missing.append("date")
            # log para debug
            print(f"[predict-match-full] Campos faltando: {missing} - raw preview: {str(raw)[:500]}")
            raise HTTPException(status_code=502, detail=f"Dados insuficientes do provedor: {missing}")

        # 3) chamar engine (seu modelo) - engine espera (home_team, away_team, date)
        # engine vai buscar features no histórico e gerar a previsão
        relatorio = engine.gerar_relatorio_json(home_team_name, away_team_name, date_iso)

        # Se engine retornar erro encapsulado no JSON, convertemos para 500
        if isinstance(relatorio, dict) and relatorio.get("status") == "ERRO":
            raise HTTPException(status_code=500, detail=relatorio.get("mensagem", "Erro interno do engine"))

        # 4) montar resposta padronizada
        response = {
            "status": "SUCESSO",
            "match_id": match_id,
            "league": {
                "id": league.get("id"),
                "name": league.get("name"),
                "season": league.get("season")
            },
            "fixture": {
                "home_team": home_team_name,
                "away_team": away_team_name,
                "date": date_iso,
                "status": fixture.get("status", {})
            },
            "previsao": relatorio,
            "meta": {
                "engine_version": "1.1.0",
                "timestamp_utc": datetime.datetime.utcnow().isoformat()
            }
        }

        if include_raw:
            response["raw"] = raw

        return response

    except HTTPException:
        raise
    except Exception as e:
        # log do erro completo para debugging
        print(f"[predict_match_full] ERRO inesperado: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# Rota debug - já existente
# ----------------------------
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


# ----------------------------
# Partida simples (já existente)
# ----------------------------
@app.get("/partida/{match_id}", tags=["Dados - API Externa"])
async def partida(match_id: int):
    dados = buscar_dados_partida(match_id)
    if not dados:
        raise HTTPException(status_code=404, detail="Partida não encontrada")
    return dados


# ----------------------------
# Rankings (já existente)
# ----------------------------
@app.get("/stats/rankings", tags=["Rankings"])
async def rankings(date: str = Query(...)):
    try:
        ranks = engine.gerar_ranking_forca(date)
        return ranks
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
