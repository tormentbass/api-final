import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb

# ==============================================================================
# CONFIGURAÇÃO DE CAMINHOS
# ==============================================================================
MODEL_PATH = "models/xgb_model.json" 
FEATURES_PATH = "models/features_finais.pkl"
HISTORICO_PATH = "models/df_historico_api.parquet"

model = None
features_finais = None
df_historico = None

def carregar_componentes():
    global model, features_finais, df_historico
    try:
        if os.path.exists(FEATURES_PATH):
            features_finais = joblib.load(FEATURES_PATH)
        if os.path.exists(HISTORICO_PATH):
            df_historico = pd.read_parquet(HISTORICO_PATH)
            df_historico["match_date"] = pd.to_datetime(df_historico["match_date"])
        
        if os.path.exists(MODEL_PATH):
            model = xgb.XGBClassifier()
            model.load_model(MODEL_PATH)
            print("✅ [ENGINE] Sistema pronto para palpites!")
            return True
        return False
    except Exception as e:
        print(f"Erro carga: {e}")
        return False

def get_latest_features(time_casa, time_fora, data_ref):
    try:
        if df_historico is None: return {"erro": "Base de dados offline"}
        
        data_ref = pd.to_datetime(data_ref)
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

        if row_c is None: return {"erro": f"Time não encontrado: {time_casa}"}
        if row_f is None: return {"erro": f"Time não encontrado: {time_fora}"}

        # --- CONSTRUÇÃO DO DICIONÁRIO DE FEATURES ---
        f_dict = {}
        
        # 1. Pega as métricas individuais (home_ e away_)
        for f in features_finais:
            if f.startswith("home_"):
                f_dict[f] = float(row_c.get(f.replace("home_", f"{pref_c}_"), 0))
            elif f.startswith("away_"):
                f_dict[f] = float(row_f.get(f.replace("away_", f"{pref_f}_"), 0))

        # 2. CALCULA AS DIFERENÇAS (O QUE ESTAVA FALTANDO!)
        for f in features_finais:
            if f.startswith("diff_"):
                metric = f.replace("diff_", "") # ex: roll_xg_for_5
                home_val = f_dict.get(f"home_{metric}", 0)
                away_val = f_dict.get(f"away_{metric}", 0)
                f_dict[f] = home_val - away_val

        # 3. Cria o DataFrame e garante a ordem das colunas
        df_ret = pd.DataFrame([f_dict])
        
        # Garante que todas as colunas do modelo existam, se alguma faltar preenche com 0
        for col in features_finais:
            if col not in df_ret.columns:
                df_ret[col] = 0.0
                
        df_ret = df_ret[features_finais] # Ordena
        
        return {
            "df_features": df_ret, 
            "net_c": f_dict.get("home_roll_xg_for_5", 0) - f_dict.get("home_roll_xg_against_5", 0), 
            "net_f": f_dict.get("away_roll_xg_for_5", 0) - f_dict.get("away_roll_xg_against_5", 0)
        }
    except Exception as e:
        return {"erro": f"Erro técnico nos dados: {str(e)}"}

def gerar_relatorio_json(time_casa, time_fora, data_ref):
    if model is None: return {"status": "ERRO", "mensagem": "Modelo desligado"}
    
    feat = get_latest_features(time_casa, time_fora, data_ref)
    if "erro" in feat: return {"status": "ERRO", "mensagem": feat["erro"]}

    try:
        probs = model.predict_proba(feat["df_features"])[0]
        res_map = {0: "Empate", 1: "Vitória Fora", 2: "Vitória Casa"}
        idx = np.argmax(probs)
        
        return {
            "status": "SUCESSO",
            "partida": f"{time_casa} vs {time_fora}",
            "previsao_final": {
                "resultado": res_map.get(idx),
                "probabilidades": {
                    "casa": f"{probs[2]*100:.2f}%",
                    "empate": f"{probs[0]*100:.2f}%",
                    "fora": f"{probs[1]*100:.2f}%"
                }
            },
            "analise_estatistica": {
                "net_xg_casa": round(float(feat["net_c"]), 2),
                "net_xg_fora": round(float(feat["net_f"]), 2)
            }
        }
    except Exception as e:
        return {"status": "ERRO", "mensagem": f"Erro na IA: {str(e)}"}

def gerar_ranking_forca(data_base, n=10):
    return {"status": "INFO", "mensagem": "Ranking disponível na próxima atualização"}

carregar_componentes()
