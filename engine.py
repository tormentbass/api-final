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
# LÓGICA DE ANÁLISE DE GOLS
# ==============================================================================
def analisar_tendencia_gols(net_c, net_f):
    expectativa_total = net_c + net_f
    if expectativa_total > 1.2:
        return "Tendência: Over 2.5 Gols 🔥"
    elif expectativa_total > 0.4:
        return "Tendência: Over 1.5 Gols 📈"
    elif expectativa_total < -0.8:
        return "Tendência: Under 2.5 Gols 🛡️"
    else:
        return "Tendência: Equilibrada (Jogo Estudado) ⚖️"

# ==============================================================================
# FUNÇÃO DE FORMATAÇÃO PARA TELEGRAM
# ==============================================================================
def gerar_texto_telegram(dados):
    p = dados["previsao_final"]
    ia_prob = p["probabilidades_ia"]
    mercado = p["probabilidades_mercado"]
    resultado = p["resultado"]
    gols = dados.get("analise_gols", "Tendência: Em análise")
    
    emoji = "🏠" if "Casa" in resultado else ("🚌" if "Fora" in resultado else "🤝")
    
    if "Casa" in resultado: chave = 'casa'
    elif "Fora" in resultado: chave = 'fora'
    else: chave = 'empate'
    
    prob_vitoria_ia = ia_prob.get(chave, "N/A")

    texto = (
        f"🎯 **PALPITE DO DIA**\n\n"
        f"⚽ **Jogo:** {dados['partida']}\n"
        f"✅ **Entrada Sugerida:** {resultado} {emoji}\n"
        f"⚽ **Mercado de Gols:** {gols}\n\n"
        f"📊 **Análise Pro-IA:**\n"
        f"🤖 Confiança da IA: {prob_vitoria_ia}\n"
    )
    
    if isinstance(mercado, dict):
        prob_m = mercado.get(chave, "N/A")
        texto += f"🏦 Probabilidade das Casas: {prob_m}\n"
    
    texto += (
        f"\n💎 **Veredito:** {dados['confianca_modelo']}\n"
        f"📈 *Análise baseada em Net xG e volume estatístico.*"
    )
    return texto

# ==============================================================================
# FUNÇÃO PARA BUSCAR ODDS
# ==============================================================================
def buscar_odds_mercado(time_casa, time_fora):
    leagues = ['soccer_epl', 'soccer_england_league1']
    try:
        for league in leagues:
            url = f'https://api.the-odds-api.com/v4/sports/{league}/odds/?apiKey={ODDS_API_KEY}&regions=eu&markets=h2h&oddsFormat=decimal'
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if not data: continue
                for jogo in data:
                    h_api, a_api = jogo['home_team'].lower(), jogo['away_team'].lower()
                    tc, tf = time_casa.lower(), time_fora.lower()
                    if tc in h_api or tf in a_api or h_api in tc or a_api in tf:
                        if not jogo['bookmakers']: continue
                        outcomes = jogo['bookmakers'][0]['markets'][0]['outcomes']
                        probs_mercado = {}
                        for o in outcomes:
                            prob = (1 / o['price']) * 100
                            if o['name'] == jogo['home_team']: 
                                probs_mercado['casa'] = f"{prob:.2f}%"
                                probs_mercado['valor_casa'] = prob
                            elif o['name'] == jogo['away_team']: 
                                probs_mercado['fora'] = f"{prob:.2f}%"
                                probs_mercado['valor_fora'] = prob
                            else: 
                                probs_mercado['empate'] = f"{prob:.2f}%"
                                probs_mercado['valor_empate'] = prob
                        return probs_mercado
        return None
    except Exception:
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
        if row_c is None or row_f is None: return {"erro": "Time não encontrado"}

        f_dict = {}
        for f in features_finais:
            if f.startswith("home_"): f_dict[f] = float(row_c.get(f.replace("home_", f"{pref_c}_"), 0))
            elif f.startswith("away_"): f_dict[f] = float(row_f.get(f.replace("away_", f"{pref_f}_"), 0))

        for f in features_finais:
            if f.startswith("diff_"):
                metric = f.replace("diff_", "")
                f_dict[f] = f_dict.get(f"home_{metric}", 0) - f_dict.get(f"away_{metric}", 0)

        df_ret = pd.DataFrame([f_dict])[features_finais]
        return {
            "df_features": df_ret, 
            "net_c": f_dict.get("home_roll_xg_for_5", 0) - f_dict.get("home_roll_xg_against_5", 0), 
            "net_f": f_dict.get("away_roll_xg_for_5", 0) - f_dict.get("away_roll_xg_against_5", 0)
        }
    except Exception as e:
        return {"erro": str(e)}

def gerar_relatorio_json(time_casa, time_fora, data_ref):
    if model is None: return {"status": "ERRO", "mensagem": "Modelo desligado"}
    feat = get_latest_features(time_casa, time_fora, data_ref)
    if "erro" in feat: return {"status": "ERRO", "mensagem": feat["erro"]}

    try:
        probs = model.predict_proba(feat["df_features"])[0]
        mercado = buscar_odds_mercado(time_casa, time_fora)
        
        pred_idx = np.argmax(probs)
        res_map = {0: "Empate", 1: "Vitória Fora", 2: "Vitória Casa"}
        resultado_ia = res_map.get(pred_idx)
        
        net_c = float(feat["net_c"])
        net_f = float(feat["net_f"])
        tendencia_gols = analisar_tendencia_gols(net_c, net_f)

        confianca = "Normal"
        if mercado:
            key_m = "valor_casa" if pred_idx == 2 else ("valor_fora" if pred_idx == 1 else "valor_empate")
            prob_m = mercado.get(key_m, 0)
            prob_ia = probs[pred_idx] * 100
            if prob_ia > 90 and prob_m < 70: confianca = "Ajustada (IA Otimista)"
            elif prob_ia > (prob_m + 15): confianca = "Alta (Valor Detectado)"
            elif prob_ia < (prob_m - 10): confianca = "Moderada (Mercado Cético)"
            mercado_limpo = {k: v for k, v in mercado.items() if not k.startswith('valor_')}
        else:
            mercado_limpo = "Indisponível"

        relatorio = {
            "status": "SUCESSO",
            "partida": f"{time_casa} vs {time_fora}",
            "confianca_modelo": confianca,
            "analise_gols": tendencia_gols,
            "previsao_final": {
                "resultado": resultado_ia,
                "probabilidades_ia": {"casa": f"{probs[2]*100:.2f}%", "empate": f"{probs[0]*100:.2f}%", "fora": f"{probs[1]*100:.2f}%"},
                "probabilidades_mercado": mercado_limpo
            },
            "estatisticas_base": {
                "net_xg_casa": round(net_c, 2),
                "net_xg_fora": round(net_f, 2)
            }
        }
        relatorio["copy_telegram"] = gerar_texto_telegram(relatorio)
        return relatorio
    except Exception as e:
        return {"status": "ERRO", "mensagem": str(e)}

# ==============================================================================
# FUNÇÃO RESTAURADA PARA EVITAR ERRO DE IMPORTAÇÃO
# ==============================================================================
def gerar_ranking_forca():
    """Retorna um ranking simples baseado no histórico carregado"""
    if df_historico is None:
        return {"status": "ERRO", "mensagem": "Base offline"}
    try:
        # Pega a data mais recente
        ultima_data = df_historico["match_date"].max()
        # Filtra os últimos 30 dias para um ranking atualizado
        recent = df_historico[df_historico["match_date"] > (ultima_data - pd.Timedelta(days=30))]
        
        ranking = recent.groupby("home_team")["home_roll_xg_for_5"].mean().sort_values(ascending=False).head(10)
        return {"status": "SUCESSO", "ranking": ranking.to_dict()}
    except:
        return {"status": "SUCESSO", "mensagem": "Ranking em processamento"}

carregar_componentes()
