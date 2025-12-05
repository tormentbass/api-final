# ==============================================================================
# ARQUIVO: engine.py (versão revisada e segura para Render)
# ==============================================================================

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import os

# Caminhos dos arquivos
MODEL_PATH = "models/xgb_model.pkl"
FEATURES_PATH = "models/features_finais.pkl"
HISTORICO_PATH = "models/df_historico_api.parquet"

model = None
features_finais = None
df_historico = None


# ==============================================================================
# 1. Carregamento dos Componentes
# ==============================================================================
def carregar_componentes():
    """Carrega modelo, features e histórico no servidor (1x)."""
    global model, features_finais, df_historico

    print("\n========== CARREGANDO COMPONENTES DE ML ==========")

    # Verifica existência dos arquivos
    for path in [MODEL_PATH, FEATURES_PATH, HISTORICO_PATH]:
        print(f"Checando arquivo: {path} ->", "OK" if os.path.exists(path) else "NÃO ENCONTRADO")

    try:
        model = joblib.load(MODEL_PATH)
        print("Modelo carregado com sucesso.")

        features_finais = joblib.load(FEATURES_PATH)
        print("Features carregadas:", len(features_finais))

        df_historico = pd.read_parquet(HISTORICO_PATH)
        df_historico["match_date"] = pd.to_datetime(df_historico["match_date"])
        print("Histórico carregado:", df_historico.shape)

        print("=================================================\n")
        return True

    except Exception as e:
        print("❌ ERRO CRÍTICO ao carregar componentes:", e)
        return False



# ==============================================================================
# 2. Extração de Features
# ==============================================================================
def get_latest_features(time_casa, time_fora, data_ref):

    if df_historico is None:
        raise Exception("Histórico não carregado.")

    data_ref = pd.to_datetime(data_ref)

    df_temp = df_historico[df_historico["match_date"] < data_ref]

    def get_last_row(team):
        home = df_temp[df_temp["home_team"] == team].sort_values("match_date", ascending=False)
        away = df_temp[df_temp["away_team"] == team].sort_values("match_date", ascending=False)

        if home.empty and away.empty:
            return None, None

        if home.empty:
            return away.iloc[0], "away"
        if away.empty:
            return home.iloc[0], "home"

        return (home.iloc[0], "home") if home.iloc[0]["match_date"] >= away.iloc[0]["match_date"] else (away.iloc[0], "away")

    row_casa, pref_casa = get_last_row(time_casa)
    row_fora, pref_fora = get_last_row(time_fora)

    if row_casa is None or row_fora is None:
        return {"erro": f"Dados insuficientes para {time_casa} ou {time_fora}."}

    # Montagem das features:
    feature_data = {}

    for feature in features_finais:
        if feature.startswith("home_roll_"):
            src = feature.replace("home_", f"{pref_casa}_")
            feature_data[feature] = row_casa.get(src, np.nan)

        elif feature.startswith("away_roll_"):
            src = feature.replace("away_", f"{pref_fora}_")
            feature_data[feature] = row_fora.get(src, np.nan)

    df_features = pd.DataFrame([feature_data])
    df_features.fillna(df_features.mean(), inplace=True)

    net_xg_casa = df_features["home_roll_xg_for_5"].iloc[0] - df_features["home_roll_xg_against_5"].iloc[0]
    net_xg_fora = df_features["away_roll_xg_for_5"].iloc[0] - df_features["away_roll_xg_against_5"].iloc[0]

    return {
        "df_features": df_features,
        "net_xg_casa": net_xg_casa,
        "net_xg_fora": net_xg_fora,
        "xg_for_casa": df_features["home_roll_xg_for_5"].iloc[0],
        "xg_against_casa": df_features["home_roll_xg_against_5"].iloc[0],
        "xg_for_fora": df_features["away_roll_xg_for_5"].iloc[0],
        "xg_against_fora": df_features["away_roll_xg_against_5"].iloc[0],
    }



# ==============================================================================
# 3. Relatório JSON
# ==============================================================================
def gerar_relatorio_json(time_casa, time_fora, data_ref):

    if model is None:
        return {"status": "ERRO", "mensagem": "Modelo não carregado."}

    feat = get_latest_features(time_casa, time_fora, data_ref)

    if "erro" in feat:
        return {"status": "ERRO", "mensagem": feat["erro"]}

    df_features = feat["df_features"]

    probabilidades = model.predict_proba(df_features)[0]
    prob_casa, prob_empate, prob_fora = [f"{p*100:.2f}%" for p in probabilidades]

    idx = np.argmax(probabilidades)
    resultado_map = {0: "Empate", 1: "Vitória Fora", 2: "Vitória Casa"}

    resultado = resultado_map[idx]
    prob_max = f"{probabilidades[idx]*100:.2f}%"

    net_c = feat["net_xg_casa"]
    net_f = feat["net_xg_fora"]
    diff = net_c - net_f

    if diff > 0.15 and idx == 2:
        justificativa = f"{time_casa} apresenta forte superioridade em Net xG ({net_c:.2f})."
    elif diff < -0.15 and idx == 1:
        justificativa = f"{time_fora} domina no Net xG ({net_f:.2f})."
    elif abs(diff) < 0.1:
        justificativa = "Confronto equilibrado estatisticamente."
    else:
        justificativa = "A previsão diverge levemente do Net xG, mas mantém coerência estatística."

    return {
        "status": "SUCESSO",
        "partida": f"{time_casa} vs {time_fora}",
        "data_base": data_ref,
        "previsao_final": {
            "resultado_provavel": resultado,
            "probabilidade_maxima": prob_max,
            "prob_casa": prob_casa,
            "prob_empate": prob_empate,
            "prob_fora": prob_fora,
        },
        "analise_estatistica": {
            "net_xg_casa": round(net_c, 2),
            "net_xg_fora": round(net_f, 2),
            "diferenca_net_xg": round(diff, 2),
        },
        "justificativa_longa": justificativa,
    }



# ==============================================================================
# 4. Batch
# ==============================================================================
def predict_batch(jogos, data_base):
    return [gerar_relatorio_json(c, f, data_base) for c, f in jogos]



# ==============================================================================
# 5. Ranking
# ==============================================================================
def gerar_ranking_forca(data_base, n=10):

    if df_historico is None:
        return {"erro": "Histórico não carregado."}

    data_ref = pd.to_datetime(data_base)
    df_f = df_historico[df_historico["match_date"] < data_ref]

    times = pd.concat([df_historico["home_team"], df_historico["away_team"]]).unique()

    ranking = []

    for t in times:
        home = df_f[df_f["home_team"] == t].sort_values("match_date", ascending=False)
        away = df_f[df_f["away_team"] == t].sort_values("match_date", ascending=False)

        if home.empty and away.empty:
            continue

        row = home.iloc[0] if (not home.empty and (away.empty or home.iloc[0]["match_date"] >= away.iloc[0]["match_date"])) else away.iloc[0]
        prefix = "home_" if row is not None and row["home_team"] == t else "away_"

        try:
            xg_for = row[f"{prefix}roll_xg_for_5"]
            xg_ag = row[f"{prefix}roll_xg_against_5"]
            goals_for = row[f"{prefix}roll_goals_for_5"]

        except Exception:
            continue

        net = xg_for - xg_ag
        perf = goals_for - xg_for

        ranking.append({
            "time": t,
            "net_xg_5": round(net, 2),
            "performance_diff_5": round(perf, 2),
            "xg_for_5": round(xg_for, 2),
        })

    if not ranking:
        return {"erro": "Dados insuficientes."}

    df_r = pd.DataFrame(ranking)

    df_forca = df_r.sort_values("net_xg_5", ascending=False).head(n).reset_index(drop=True)
    df_valor = df_r.sort_values("performance_diff_5", ascending=False).head(n).reset_index(drop=True)

    df_forca["rank"] = df_forca.index + 1
    df_valor["rank"] = df_valor.index + 1

    return {
        "data_referencia": data_base,
        "ranking_forca": df_forca.to_dict("records"),
        "ranking_valor": df_valor.to_dict("records"),
    }


# Carrega na importação
carregar_componentes()
