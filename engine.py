# ====================================================================================================
# ARQUIVO: engine.py
# VERSÃO: 1.2.0 - FIX VERSÃO ML
# DATA: 24/02/2026
# DESCRIÇÃO: Motor ML com correção de compatibilidade para XGBoost antigo
# ====================================================================================================

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# ====================================================================================================
# PATHS
# ====================================================================================================
MODEL_PATH = "models/xgb_model.pkl"
FEATURES_PATH = "models/features_finais.pkl"
HISTORICO_PATH = "models/df_historico_api.parquet"

# ====================================================================================================
# OBJETOS GLOBAIS
# ====================================================================================================
model = None
features_finais = None
df_historico = None

# ====================================================================================================
# CARREGAMENTO DE COMPONENTES (COM FIX DE VERSÃO)
# ====================================================================================================
def carregar_componentes():
    global model, features_finais, df_historico

    print("\n========== [ENGINE] CARREGANDO COMPONENTES ==========")
    for path in [MODEL_PATH, FEATURES_PATH, HISTORICO_PATH]:
        print(f"[CHECK] {path}: {'OK' if os.path.exists(path) else 'NÃO ENCONTRADO'}")

    # 1. Carregar Features e Histórico
    try:
        features_finais = joblib.load(FEATURES_PATH)
        df_historico = pd.read_parquet(HISTORICO_PATH)
        df_historico["match_date"] = pd.to_datetime(df_historico["match_date"])
        print(f"[ENGINE] Histórico e Features carregados com sucesso.")
    except Exception as e:
        print(f"❌ ERRO NOS DADOS: {e}")
        return False

    # 2. Carregar Modelo com Fix para 'use_label_encoder'
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print("[ENGINE] Modelo ML carregado com sucesso!")
    except Exception as e:
        if "use_label_encoder" in str(e):
            print("[ENGINE] Detectado erro de versão XGBoost. Aplicando correção...")
            try:
                # Técnica de correção: Carregamos o objeto e removemos o atributo fantasma
                import xgboost as xgb
                raw_model = joblib.load(MODEL_PATH)
                if hasattr(raw_model, 'use_label_encoder'):
                    delattr(raw_model, 'use_label_encoder')
                model = raw_model
                print("[ENGINE] Modelo carregado com sucesso após correção de versão!")
            except Exception as e2:
                print(f"❌ FALHA NA CORREÇÃO DO MODELO: {e2}")
                model = None
        else:
            print(f"❌ ERRO DESCONHECIDO NO MODELO: {e}")
            model = None

    print("========== [ENGINE] STATUS FINAL: {'PRONTO' if model else 'FALHA'} ==========\n")
    return True if model else False

# ====================================================================================================
# EXTRAÇÃO DE FEATURES
# ====================================================================================================
def get_latest_features(time_casa, time_fora, data_ref):
    if df_historico is None:
        raise RuntimeError("Histórico não carregado pelo engine.")

    try:
        data_ref = pd.to_datetime(data_ref, errors='coerce')
        if pd.isna(data_ref):
            return {"erro": f"Data inválida: {data_ref}"}
    except Exception as e:
        return {"erro": f"Erro ao converter data: {e}"}

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

    feature_data = {}
    for feature in features_finais:
        try:
            if feature.startswith("home_"):
                src = feature.replace("home_", f"{pref_casa}_")
                feature_data[feature] = row_casa.get(src, 0)
            elif feature.startswith("away_"):
                src = feature.replace("away_", f"{pref_fora}_")
                feature_data[feature] = row_fora.get(src, 0)
        except Exception:
            feature_data[feature] = 0

    df_features = pd.DataFrame([feature_data])
    df_features.fillna(0, inplace=True)

    net_xg_casa = df_features.get("home_roll_xg_for_5", [0])[0] - df_features.get("home_roll_xg_against_5", [0])[0]
    net_xg_fora = df_features.get("away_roll_xg_for_5", [0])[0] - df_features.get("away_roll_xg_against_5", [0])[0]

    return {
        "df_features": df_features,
        "net_xg_casa": net_xg_casa,
        "net_xg_fora": net_xg_fora,
        "xg_for_casa": df_features.get("home_roll_xg_for_5", [0])[0],
        "xg_against_casa": df_features.get("home_roll_xg_against_5", [0])[0],
        "xg_for_fora": df_features.get("away_roll_xg_for_5", [0])[0],
        "xg_against_fora": df_features.get("away_roll_xg_against_5", [0])[0],
    }

# ====================================================================================================
# RELATÓRIO JSON
# ====================================================================================================
def gerar_relatorio_json(time_casa, time_fora, data_ref):
    if model is None:
        return {"status": "ERRO", "mensagem": "Modelo não carregado."}

    feat = get_latest_features(time_casa, time_fora, data_ref)
    if "erro" in feat:
        return {"status": "ERRO", "mensagem": feat["erro"]}

    df_features = feat["df_features"]

    try:
        # Garante que as colunas estejam na ordem correta que o modelo espera
        df_features = df_features[features_finais]
        prob = model.predict_proba(df_features)[0]
        
        prob_casa, prob_empate, prob_fora = [f"{p*100:.2f}%" for p in prob]
        idx = np.argmax(prob)
        resultado_map = {0: "Empate", 1: "Vitória Fora", 2: "Vitória Casa"}
        resultado = resultado_map.get(idx, "Empate")
    except Exception as e:
        return {"status": "ERRO", "mensagem": f"Erro na predição ML: {e}"}

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
        justificativa = "Previsão coerente, porém divergindo levemente do Net xG."

    return {
        "status": "SUCESSO",
        "partida": f"{time_casa} vs {time_fora}",
        "data_base": str(data_ref.date()) if hasattr(data_ref, 'date') else str(data_ref),
        "previsao_final": {
            "resultado_provavel": resultado,
            "probabilidades": {
                "casa": prob_casa,
                "empate": prob_empate,
                "fora": prob_fora,
            },
        },
        "analise_estatistica": {
            "net_xg_casa": round(net_c, 2),
            "net_xg_fora": round(net_f, 2),
            "diff_net_xg": round(diff, 2),
        },
        "justificativa": justificativa,
    }

# ====================================================================================================
# RANKING E BATCH
# ====================================================================================================
def predict_batch(jogos, data_base):
    return [gerar_relatorio_json(c, f, data_base) for c, f in jogos]

def gerar_ranking_forca(data_base, n=10):
    if df_historico is None:
        return {"erro": "Histórico não carregado."}

    data_ref = pd.to_datetime(data_base, errors='coerce')
    if pd.isna(data_ref):
        return {"erro": f"Data inválida: {data_base}"}

    df_f = df_historico[df_historico["match_date"] < data_ref]
    times = pd.concat([df_historico["home_team"], df_historico["away_team"]]).unique()
    ranking = []

    for t in times:
        home = df_f[df_f["home_team"] == t].sort_values("match_date", ascending=False)
        away = df_f[df_f["away_team"] == t].sort_values("match_date", ascending=False)

        if home.empty and away.empty: continue

        row = home.iloc[0] if (not home.empty and (away.empty or home.iloc[0]["match_date"] >= away.iloc[0]["match_date"])) else away.iloc[0]
        prefix = "home_" if row["home_team"] == t else "away_"

        try:
            xg_for = row.get(f"{prefix}roll_xg_for_5", 0)
            xg_ag = row.get(f"{prefix}roll_xg_against_5", 0)
            goals_for = row.get(f"{prefix}roll_goals_for_5", 0)
        except: continue

        ranking.append({
            "time": t,
            "net_xg_5": round(xg_for - xg_ag, 2),
            "performance_diff_5": round(goals_for - xg_for, 2),
            "xg_for_5": round(xg_for, 2),
        })

    if not ranking: return {"erro": "Dados insuficientes."}

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

# ====================================================================================================
# CARREGAMENTO AUTOMÁTICO
# ====================================================================================================
carregar_componentes()
