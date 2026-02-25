import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb

MODEL_PATH = "models/xgb_model.json" 
FEATURES_PATH = "models/features_finais.pkl"
HISTORICO_PATH = "models/df_historico_api.parquet"

model = None
features_finais = None
df_historico = None

def carregar_componentes():
    global model, features_finais, df_historico
    try:
        features_finais = joblib.load(FEATURES_PATH)
        df_historico = pd.read_parquet(HISTORICO_PATH)
        df_historico["match_date"] = pd.to_datetime(df_historico["match_date"])
        
        if os.path.exists(MODEL_PATH):
            model = xgb.XGBClassifier()
            model.load_model(MODEL_PATH)
            return True
        return False
    except Exception as e:
        print(f"Erro carga: {e}")
        return False

def get_latest_features(time_casa, time_fora, data_ref):
    try:
        if df_historico is None: return {"erro": "Histórico não carregado"}
        
        data_ref = pd.to_datetime(data_ref)
        # Filtra apenas jogos ANTES da data da aposta
        df_temp = df_historico[df_historico["match_date"] < data_ref].copy()

        def get_last_row(team):
            h = df_temp[df_temp["home_team"] == team].sort_values("match_date", ascending=False)
            a = df_temp[df_temp["away_team"] == team].sort_values("match_date", ascending=False)
            
            if h.empty and a.empty: return None, None
            if h.empty: return a.iloc[0], "away"
            if a.empty: return h.iloc[0], "home"
            
            return (h.iloc[0], "home") if h.iloc[0]["match_date"] >= a.iloc[0]["match_date"] else (a.iloc[0], "away")

        row_c, pref_c = get_last_row(time_casa)
        row_f, pref_f = get_last_row(time_fora)

        if row_c is None: return {"erro": f"Time Casa [{time_casa}] não encontrado no histórico"}
        if row_f is None: return {"erro": f"Time Fora [{time_fora}] não encontrado no histórico"}

        f_dict = {}
        for f in features_finais:
            if f.startswith("home_"):
                f_dict[f] = row_c.get(f.replace("home_", f"{pref_c}_"), 0)
            elif f.startswith("away_"):
                f_dict[f] = row_f.get(f.replace("away_", f"{pref_f}_"), 0)
        
        df_ret = pd.DataFrame([f_dict])[features_finais].fillna(0)
        return {"df_features": df_ret, "net_c": f_dict.get("home_roll_xg_for_5", 0) - f_dict.get("home_roll_xg_against_5", 0), "net_f": f_dict.get("away_roll_xg_for_5", 0) - f_dict.get("away_roll_xg_against_5", 0)}
    except Exception as e:
        return {"erro": f"Erro no processamento de dados: {str(e)}"}

def gerar_relatorio_json(time_casa, time_fora, data_ref):
    if model is None: return {"status": "ERRO", "mensagem": "Modelo não carregado"}
    
    feat = get_latest_features(time_casa, time_fora, data_ref)
    if "erro" in feat: return {"status": "ERRO", "mensagem": feat["erro"]}

    try:
        # Tenta predição
        probs = model.predict_proba(feat["df_features"])[0]
        res_map = {0: "Empate", 1: "Vitória Fora", 2: "Vitória Casa"}
        
        return {
            "status": "SUCESSO",
            "partida": f"{time_casa} vs {time_fora}",
            "previsao_final": {
                "resultado": res_map.get(np.argmax(probs)),
                "probabilidades": {"casa": f"{probs[2]*100:.2f}%", "empate": f"{probs[0]*100:.2f}%", "fora": f"{probs[1]*100:.2f}%"}
            }
        }
    except Exception as e:
        return {"status": "ERRO", "mensagem": f"Erro na IA: {str(e)}"}

def gerar_ranking_forca(data_base, n=10):
    return {"status": "OK", "mensagem": "Ranking desativado para teste"}

carregar_componentes()
