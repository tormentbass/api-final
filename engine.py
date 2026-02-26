import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb
import requests

# ==============================================================================
# CONFIGURAÇÃO DE CAMINHOS E CHAVES
# ==============================================================================
MODEL_PATH = "models/xgb_model.json" 
FEATURES_PATH = "models/features_finais.pkl"
HISTORICO_PATH = "models/df_historico_api.parquet"
ODDS_API_KEY = "179289ff6d63366f8af6b9de37fd9d7e"

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
        print(f"❌ Erro carga: {e}")
        return False

# ==============================================================================
# FUNÇÃO PARA BUSCAR ODDS EM TEMPO REAL (MELHORADA)
# ==============================================================================
def buscar_odds_mercado(time_casa, time_fora):
    """Consulta a The Odds API focando na Premier League (plano free)"""
    # soccer_epl ou soccer_england_league1 são os códigos padrão para Premier League
    leagues = ['soccer_epl', 'soccer_england_league1']
    
    try:
        for league in leagues:
            # Adicionado oddsFormat=decimal para facilitar a vida
            url = f'https://api.the-odds-api.com/v4/sports/{league}/odds/?apiKey={ODDS_API_KEY}&regions=eu&markets=h2h&oddsFormat=decimal'
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if not data: continue
                
                for jogo in data:
                    h_api = jogo['home_team'].lower()
                    a_api = jogo['away_team'].lower()
                    tc = time_casa.lower()
                    tf = time_fora.lower()

                    # Verifica se um dos nomes bate (contém o termo buscado)
                    if tc in h_api or tf in a_api or h_api in tc or a_api in tf:
                        if not jogo['bookmakers']: continue
                        
                        # Pegamos a primeira casa de apostas disponível
                        outcomes = jogo['bookmakers'][0]['markets'][0]['outcomes']
                        probs_mercado = {}
                        
                        for o in outcomes:
                            prob = (1 / o['price']) * 100
                            # Mapeia os nomes da API de volta para nossas chaves
                            if o['name'] == jogo['home_team']: 
                                probs_mercado['casa'] = f"{prob:.2f}%"
                                probs_mercado['valor_casa'] = prob # p/ cálculo de confiança
                            elif o['name'] == jogo['away_team']: 
                                probs_mercado['fora'] = f"{prob:.2f}%"
                                probs_mercado['valor_fora'] = prob
                            else: 
                                probs_mercado['empate'] = f"{prob:.2f}%"
                                probs_mercado['valor_empate'] = prob
                        return probs_mercado
        return None
    except Exception as e:
        print(f"⚠️ Erro API Odds: {e}")
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
        
        # 2. Busca Odds Reais
        mercado = buscar_odds_mercado(time_casa, time_fora)
        
        # 3. Lógica de Confiança Calibrada
        confianca = "Normal"
        pred_idx = np.argmax(probs)
        res_map = {0: "Empate", 1: "Vitória Fora", 2: "Vitória Casa"}
        resultado_ia = res_map.get(pred_idx)

        if mercado:
            # Pega a probabilidade numérica do mercado para o resultado que a IA escolheu
            key_m = "valor_casa" if pred_idx == 2 else ("valor_fora" if pred_idx == 1 else "valor_empate")
            prob_m = mercado.get(key_m, 0)
            
            prob_ia_atual = probs[pred_idx] * 100
            
            # Filtros de segurança
            if prob_ia_atual > 90 and prob_m < 70:
                confianca = "Ajustada (IA Otimista)"
            elif prob_ia_atual > (prob_m + 15):
                confianca = "Alta (Valor Detectado)"
            elif prob_ia_atual < (prob_m - 10):
                confianca = "Moderada (Mercado Cético)"
            
            # Limpa os valores auxiliares para o JSON não ficar sujo
            mercado_limpo = {k: v for k, v in mercado.items() if not k.startswith('valor_')}
        else:
            mercado_limpo = "Indisponível (API/Fora da Premier League)"

        return {
            "status": "SUCESSO",
            "partida": f"{time_casa} vs {time_fora}",
            "confianca_modelo": confianca,
            "previsao_final": {
                "resultado": resultado_ia,
                "probabilidades_ia": {
                    "casa": f"{prob_casa_ia:.2f}%",
                    "empate": f"{prob_empate_ia:.2f}%",
                    "fora": f"{prob_fora_ia:.2f}%"
                },
                "probabilidades_mercado": mercado_limpo
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
