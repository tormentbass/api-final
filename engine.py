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
    print("\n========== [ENGINE] CARREGANDO COMPONENTES ==========")
    
    try:
        # 1. Carregar Features e Histórico
        if os.path.exists(FEATURES_PATH):
            features_finais = joblib.load(FEATURES_PATH)
            print("✅ Features OK")
        if os.path.exists(HISTORICO_PATH):
            df_historico = pd.read_parquet(HISTORICO_PATH)
            df_historico["match_date"] = pd.to_datetime(df_historico["match_date"])
            print("✅ Histórico OK")
        
        # 2. Carregar o Modelo JSON (Correção para o erro de Cast)
        if os.path.exists(MODEL_PATH):
            # Usamos o XGBClassifier em vez do Booster puro para evitar o erro de cast no Render
            model = xgb.XGBClassifier()
            model.load_model(MODEL_PATH)
            print("✅ Modelo JSON carregado!")
        else:
            print(f"❌ Arquivo não encontrado: {MODEL_PATH}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Erro no carregamento: {e}")
        return False

def get_latest_features(time_casa, time_fora, data_ref):
    if df_historico is None: return {"erro": "Histórico offline"}
    
    data_ref = pd.to_datetime(data_ref)
    df_temp = df_historico[df_historico["match_date"] < data_ref]

    def get_last_row(team):
        h = df_temp[df_temp["home_team"] == team].sort_values("match_date", ascending=False)
        a = df_temp[df_temp["away_team"] == team].sort_values("match_date", ascending=False)
        if h.empty and a.empty: return None, None
        if h.empty: return a.iloc[0], "away"
        if a.empty: return h.iloc[0], "home"
        return (h.iloc[0], "home") if h.iloc[0]["match_date"] >= a.iloc[0]["match_date"] else (a.iloc[0], "away")

    row_c, pref_c = get_last_row(time_casa)
    row_f, pref_f = get_last_row(time_fora)

    if row_c is None or row_f is None:
        return {"erro": "Dados insuficientes no histórico"}

    f_dict = {}
    for f in features_finais:
        if f.startswith("home_"):
            f_dict[f] = row_c.get(f.replace("home_", f"{pref_c}_"), 0)
        elif f.startswith("away_"):
            f_dict[f] = row_f.get(f.replace("away_", f"{pref_f}_"), 0)
    
    df_ret = pd.DataFrame([f_dict])[features_finais].fillna(0)
    return {
        "df_features": df_ret, 
        "net_c": f_dict.get("home_roll_xg_for_5", 0) - f_dict.get("home_roll_xg_against_5", 0), 
        "net_f": f_dict.get("away_roll_xg_for_5", 0) - f_dict.get("away_roll_xg_against_5", 0)
    }

def gerar_relatorio_json(time_casa, time_fora, data_ref):
    if model is None: return {"status": "ERRO", "mensagem": "Modelo não carregado"}
    
    feat = get_latest_features(time_casa, time_fora, data_ref)
    if "erro" in feat: return {"status": "ERRO", "mensagem": feat["erro"]}

    try:
        # Voltamos para o predict_proba que funciona com XGBClassifier
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
        return {"status": "ERRO", "mensagem": str(e)}

# FUNÇÃO ADICIONADA PARA EVITAR O ERRO DE IMPORTAÇÃO NO APP.PY
def gerar_ranking_forca(data_base, n=10):
    return {"status": "OK", "mensagem": "Ranking disponível em breve"}

# Inicializa
carregar_componentes()
