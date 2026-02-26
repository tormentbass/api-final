import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb
import requests  # Import necessário para a API de Odds

# ==============================================================================
# CONFIGURAÇÃO DE CAMINHOS E CHAVES
# ==============================================================================
MODEL_PATH = "models/xgb_model.json" 
FEATURES_PATH = "models/features_finais.pkl"
HISTORICO_PATH = "models/df_historico_api.parquet"
ODDS_API_KEY = "179289ff6d63366f8af6b9de37fd9d7e" # Sua chave ativa

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

# ==============================================================================
# FUNÇÃO PARA BUSCAR ODDS EM TEMPO REAL
# ==============================================================================
def buscar_odds_mercado(time_casa, time_fora):
    """Consulta a The Odds API para pegar a probabilidade implícita das casas"""
    # Lista de ligas para buscar (Premier League, etc). Pode ser ajustado conforme a necessidade.
    leagues = ['soccer_england_league1', 'soccer_brazil_campeonato_brasileiro', 'soccer_uefa_champs_league']
    
    try:
        for league in leagues:
            url = f'https://api.the-odds-api.com/v4/sports/{league}/odds/?apiKey={ODDS_API_KEY}&regions=eu&markets=h2h'
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                for jogo in data:
                    # Tenta encontrar o jogo por nome aproximado
                    if time_casa.lower() in jogo['home_team'].lower() or time_fora.lower() in jogo['away_team'].lower():
                        if not jogo['bookmakers']: continue
                        
                        outcomes = jogo['bookmakers'][0]['markets'][0]['outcomes']
                        probs_mercado = {}
                        for o in outcomes:
                            # Converte decimal em %
                            prob = (1 / o['price']) * 100
                            if o['name'] == jogo['home_team']: probs_mercado['casa'] = prob
                            elif o['name'] == jogo['away_team']: probs_mercado['fora'] = prob
                            else: probs_mercado['empate'] = prob
                        return probs_mercado
        return None
    except:
        return None

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

        f_dict = {}
        for f in features_finais:
            if f.startswith("home_"):
                f_dict[f] = float(row_c.get(f.replace("home_", f"{pref_c}_"), 0))
            elif f.startswith("away_"):
                f_dict[f] = float(row_f.get(f.replace("away_", f"{pref_f}_"), 0))

        for f in features_finais:
            if f.startswith("diff_"):
                metric = f.replace("diff_", "")
                home_val = f_dict.get(f"home_{metric}", 0)
                away_val = f_dict.get(f"away_{metric}", 0)
                f_dict[f] = home_val - away_val

        df_ret = pd.DataFrame([f_dict])
        for col in features_finais:
            if col not in df_ret.columns:
                df_ret[col] = 0.0
                
        df_ret = df_ret[features_finais] 
        
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
        # 1. Probabilidades da IA
        probs = model.predict_proba(feat["df_features"])[0]
        prob_casa_ia = probs[2] * 100
        prob_empate_ia = probs[0] * 100
        prob_fora_ia = probs[1] * 100
        
        # 2. Busca Odds Reais para calibração
        mercado = buscar_odds_mercado(time_casa, time_fora)
        
        # Lógica de Confiança (Se a IA for muito longe do mercado, a gente 'avisa')
        confianca = "Normal"
        if mercado:
            # Pega a probabilidade do resultado previsto pela IA no mercado
            pred_idx = np.argmax(probs)
            key_mercado = "casa" if pred_idx == 2 else ("fora" if pred_idx == 1 else "empate")
            prob_m = mercado.get(key_mercado, 0)
            
            # Se a IA der > 90% mas o mercado < 70%, reduzimos a confiança exibida
            if (probs[pred_idx] * 100) > 90 and prob_m < 70:
                confianca = "Ajustada (IA Otimista)"
            elif (probs[pred_idx] * 100) > prob_m + 15:
                confianca = "Alta (Valor Detectado)"

        res_map = {0: "Empate", 1: "Vitória Fora", 2: "Vitória Casa"}
        idx = np.argmax(probs)
        
        return {
            "status": "SUCESSO",
            "partida": f"{time_casa} vs {time_fora}",
            "confianca_modelo": confianca,
            "previsao_final": {
                "resultado": res_map.get(idx),
                "probabilidades_ia": {
                    "casa": f"{prob_casa_ia:.2f}%",
                    "empate": f"{prob_empate_ia:.2f}%",
                    "fora": f"{prob_fora_ia:.2f}%"
                },
                "probabilidades_mercado": mercado if mercado else "Indisponível (API)"
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
