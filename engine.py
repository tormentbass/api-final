# ====================================================================================================
# ARQUIVO: engine.py
# VERSÃO: 1.3.0 - ULTRA ROBUST LOAD
# DATA: 24/02/2026
# DESCRIÇÃO: Motor ML com bypass de erro de versão do XGBoost
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
# CARREGAMENTO DE COMPONENTES (BYPASS DE VERSÃO)
# ====================================================================================================
def carregar_componentes():
    global model, features_finais, df_historico

    print("\n========== [ENGINE] INICIANDO CARREGAMENTO CRÍTICO ==========")
    
    # 1. Carregar Dados de Apoio (Features e Parquet)
    try:
        if os.path.exists(FEATURES_PATH):
            features_finais = joblib.load(FEATURES_PATH)
            print(f"[OK] Features carregadas: {len(features_finais)}")
        
        if os.path.exists(HISTORICO_PATH):
            df_historico = pd.read_parquet(HISTORICO_PATH)
            df_historico["match_date"] = pd.to_datetime(df_historico["match_date"])
            print(f"[OK] Histórico carregado: {len(df_historico)} linhas")
    except Exception as e:
        print(f"❌ ERRO NOS DADOS: {e}")

    # 2. Carregar Modelo com Técnica de Limpeza de Buffer
    if os.path.exists(MODEL_PATH):
        try:
            print(f"[INFO] Tentando carregar modelo: {MODEL_PATH}")
            # Tentativa 1: Carregamento Padrão
            model = joblib.load(MODEL_PATH)
            
            # Se carregou, mas tem o atributo antigo, desativa aqui
            if hasattr(model, 'use_label_encoder'):
                try:
                    model.set_params(use_label_encoder=False)
                except:
                    delattr(model, 'use_label_encoder')
            
            print("[SUCCESS] Modelo carregado via Joblib!")
            
        except Exception as e:
            print(f"[WARN] Falha no carregamento padrão: {e}")
            print("[ACTION] Tentando Fallback: Limpeza de atributos antigos...")
            
            try:
                import xgboost as xgb
                # Tentativa 2: Forçar remoção de atributos incompatíveis durante o load
                # Isso funciona em algumas versões onde o joblib falha no mapeamento
                raw_obj = joblib.load(MODEL_PATH)
                
                # Se for um modelo XGBoost, removemos o lixo da versão anterior
                if 'XGB' in str(type(raw_obj)):
                    if hasattr(raw_obj, 'use_label_encoder'):
                        delattr(raw_obj, 'use_label_encoder')
                    model = raw_obj
                    print("[SUCCESS] Modelo carregado com Limpeza de Atributos!")
            except Exception as e2:
                print(f"❌ FALHA CRÍTICA: O arquivo .pkl é incompatível com este servidor Linux/Render.")
                print(f"ERRO: {e2}")
                model = None

    print(f"========== [ENGINE] STATUS: {'PRONTO' if model is not None else 'ERRO'} ==========\n")
    return True if model else False

# ====================================================================================================
# EXTRAÇÃO DE FEATURES
# ====================================================================================================
def get_latest_features(time_casa, time_fora, data_ref):
    if df_historico is None:
        return {"erro": "Histórico não carregado no servidor."}

    try:
        data_ref = pd.to_datetime(data_ref)
    except:
        return {"erro": "Formato de data inválido. Use AAAA-MM-DD."}

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
        return {"erro": f"Times não encontrados no histórico: {time_casa}/{time_fora}"}

    f_dict = {}
    for f in features_finais:
        if f.startswith("home_"):
            f_dict[f] = row_c.get(f.replace("home_", f"{pref_c}_"), 0)
        elif f.startswith("away_"):
            f_dict[f] = row_f.get(f.replace("away_", f"{pref_f}_"), 0)
    
    df_ret = pd.DataFrame([f_dict])[features_finais].fillna(0)
    
    return {
        "df_features": df_ret,
        "net_xg_casa": f_dict.get("home_roll_xg_for_5", 0) - f_dict.get("home_roll_xg_against_5", 0),
        "net_xg_fora": f_dict.get("away_roll_xg_for_5", 0) - f_dict.get("away_roll_xg_against_5", 0)
    }

# ====================================================================================================
# RELATÓRIO JSON
# ====================================================================================================
def gerar_relatorio_json(time_casa, time_fora, data_ref):
    if model is None:
        return {"status": "ERRO", "mensagem": "Modelo ML não disponível no servidor."}

    feat = get_latest_features(time_casa, time_fora, data_ref)
    if "erro" in feat:
        return {"status": "ERRO", "mensagem": feat["erro"]}

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
                "net_xg_casa": round(float(feat["net_xg_casa"]), 2),
                "net_xg_fora": round(float(feat["net_xg_fora"]), 2)
            }
        }
    except Exception as e:
        return {"status": "ERRO", "mensagem": f"Erro na execução do modelo: {e}"}

# ====================================================================================================
# RANKINGS
# ====================================================================================================
def gerar_ranking_forca(data_base, n=10):
    if df_historico is None: return {"erro": "Sem dados."}
    try:
        d_ref = pd.to_datetime(data_base)
        df_f = df_historico[df_historico["match_date"] < d_ref]
        times = pd.concat([df_f["home_team"], df_f["away_team"]]).unique()
        res = []
        for t in times:
            h = df_f[df_f["home_team"] == t].sort_values("match_date", ascending=False)
            a = df_f[df_f["away_team"] == t].sort_values("match_date", ascending=False)
            row = h.iloc[0] if (not h.empty and (a.empty or h.iloc[0]["match_date"] >= a.iloc[0]["match_date"])) else a.iloc[0]
            p = "home_" if row["home_team"] == t else "away_"
            res.append({"time": t, "net_xg_5": round(row.get(f"{p}roll_xg_for_5", 0) - row.get(f"{p}roll_xg_against_5", 0), 2)})
        
        final = pd.DataFrame(res).sort_values("net_xg_5", ascending=False).head(n).to_dict("records")
        return {"ranking": final}
    except:
        return {"erro": "Falha ao gerar ranking."}

# Inicializa
carregar_componentes()
