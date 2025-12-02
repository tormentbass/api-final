# ==============================================================================
# ARQUIVO: engine.py
# CONTEÚDO: Funções de ML, Ranking e Carregamento de Modelo
# ==============================================================================

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import os

# --- 1. CONFIGURAÇÃO E CARREGAMENTO (Caminhos RELATIVOS para o servidor) ---
MODEL_PATH = "models/xgb_model.pkl"
FEATURES_PATH = "models/features_finais.pkl"
HISTORICO_PATH = "models/df_historico_api.parquet" # Seu df_base otimizado

model = None
features_finais = None
df_historico = None

def carregar_componentes():
    """Carrega o modelo, features e histórico na memória da API."""
    global model, features_finais, df_historico
    try:
        model = joblib.load(MODEL_PATH)
        with open(FEATURES_PATH, 'rb') as f:
            features_finais = joblib.load(f)
        
        df_historico = pd.read_parquet(HISTORICO_PATH)
        df_historico['match_date'] = pd.to_datetime(df_historico['match_date'])
        
        print("✅ Componentes de ML carregados com sucesso.")
        return True
    except FileNotFoundError as e:
        # Se você tiver esse erro no servidor, verifique se a pasta 'models' foi carregada corretamente.
        print(f"❌ Erro ao carregar componente. Verifique o caminho: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro desconhecido ao carregar componentes: {e}")
        return False

# --- 2. FUNÇÃO: EXTRAÇÃO DE FEATURES PARA PREVISÃO ---

def get_latest_features(time_casa, time_fora, data_ref):
    """
    Extrai as rolling features mais recentes para a partida dada.
    """
    if df_historico is None:
        raise Exception("Histórico de dados não carregado.")
    
    data_ref = pd.to_datetime(data_ref)
    
    # 1. Filtra dados anteriores
    df_temp = df_historico[df_historico['match_date'] < data_ref]
    
    # 2. Localiza a última linha para cada time
    def get_last_row(team):
        home_rows = df_temp[df_temp['home_team'] == team].sort_values('match_date', ascending=False)
        away_rows = df_temp[df_temp['away_team'] == team].sort_values('match_date', ascending=False)
        
        if home_rows.empty and away_rows.empty:
            return None, None
        
        if home_rows.empty:
            return away_rows.iloc[0], 'away'
        if away_rows.empty:
            return home_rows.iloc[0], 'home'
            
        last_home = home_rows.iloc[0]
        last_away = away_rows.iloc[0]
        
        if last_home['match_date'] >= last_away['match_date']:
            return last_home, 'home'
        else:
            return last_away, 'away'
    
    last_row_casa, prefix_casa = get_last_row(time_casa)
    last_row_fora, prefix_fora = get_last_row(time_fora)
    
    if last_row_casa is None or last_row_fora is None:
        return {"erro": f"Dados insuficientes para {time_casa} ou {time_fora}."}

    # 3. Mapeia as features
    feature_data = {}
    for feature in features_finais:
        if feature.startswith('home_roll_'):
            original_prefix = 'home_' if prefix_casa == 'home' else 'away_'
            source_col = feature.replace('home_', f'{original_prefix}')
            feature_data[feature] = last_row_casa.get(source_col, np.nan)
        elif feature.startswith('away_roll_'):
            original_prefix = 'home_' if prefix_fora == 'home' else 'away_'
            source_col = feature.replace('away_', f'{original_prefix}')
            feature_data[feature] = last_row_fora.get(source_col, np.nan)

    # 4. Converte para DataFrame e preenche NaNs
    df_features = pd.DataFrame([feature_data], columns=features_finais)
    df_features.fillna(df_features.mean(), inplace=True) 
    
    # 5. Retorna o DF para o modelo e o Net xG para a justificativa
    net_xg_casa = df_features['home_roll_xg_for_5'].iloc[0] - df_features['home_roll_xg_against_5'].iloc[0]
    net_xg_fora = df_features['away_roll_xg_for_5'].iloc[0] - df_features['away_roll_xg_against_5'].iloc[0]
    
    return {
        "df_features": df_features,
        "net_xg_casa": net_xg_casa,
        "net_xg_fora": net_xg_fora,
        "xg_for_casa": df_features['home_roll_xg_for_5'].iloc[0],
        "xg_against_casa": df_features['home_roll_xg_against_5'].iloc[0],
        "xg_for_fora": df_features['away_roll_xg_for_5'].iloc[0],
        "xg_against_fora": df_features['away_roll_xg_against_5'].iloc[0],
    }

# --- 3. FUNÇÃO: GERAÇÃO DE RELATÓRIO JSON ---

def gerar_relatorio_json(time_casa, time_fora, data_ref):
    """Gera um relatório JSON completo de previsão e análise."""
    if model is None:
        return {"status": "ERRO", "mensagem": "Modelo de IA não carregado."}

    feature_result = get_latest_features(time_casa, time_fora, data_ref)
    
    if "erro" in feature_result:
        return {"status": "ERRO", "mensagem": feature_result["erro"]}
        
    df_features = feature_result["df_features"]

    probabilidades = model.predict_proba(df_features)[0]
    prob_casa, prob_draw, prob_fora = [f"{p*100:.2f}%" for p in probabilidades]
    
    resultado_map = {0: "Empate", 1: "Vitória Fora", 2: "Vitória Casa"}
    predicao_index = np.argmax(probabilidades)
    resultado_provavel = resultado_map[predicao_index]
    prob_max = f"{probabilidades[predicao_index]*100:.2f}%"

    net_xg_casa = feature_result["net_xg_casa"]
    net_xg_fora = feature_result["net_xg_fora"]
    diff_net_xg = net_xg_casa - net_xg_fora

    if diff_net_xg > 0.15 and predicao_index == 2:
        justificativa_texto = f"Forte tendência de vitória do {time_casa}. Sua Força Líquida (Net xG de {net_xg_casa:.2f}) é significativamente superior ao {time_fora}."
    elif diff_net_xg < -0.15 and predicao_index == 1:
        justificativa_texto = f"O {time_fora} é o favorito. Sua Força Líquida (Net xG de {net_xg_fora:.2f}) domina o confronto, validando a previsão da IA."
    elif abs(diff_net_xg) < 0.1:
        justificativa_texto = "Confronto de alta equivalência estatística (Net xG). A previsão é baseada em margens estreitas e tendências históricas do modelo."
    else:
        justificativa_texto = "O modelo de IA fez uma previsão que diverge ligeiramente do Net xG. A diferença é pequena, mas a confiança na previsão é mantida."
    
    relatorio = {
        "status": "SUCESSO",
        "partida": f"{time_casa} vs {time_fora}",
        "data_base": data_ref,
        "previsao_final": {
            "resultado_provavel": resultado_provavel,
            "probabilidade_maxima": prob_max,
            "prob_casa": prob_casa,
            "prob_empate": prob_draw,
            "prob_fora": prob_fora,
        },
        "analise_estatistica": {
            "metrica_chave": "Net xG (Últ. 5 Jogos)",
            "time_casa": {"net_xg": round(net_xg_casa, 2), "xg_for_5": round(feature_result["xg_for_casa"], 2)},
            "time_fora": {"net_xg": round(net_xg_fora, 2), "xg_for_5": round(feature_result["xg_for_fora"], 2)},
            "diferenca_net_xg": round(diff_net_xg, 2),
        },
        "justificativa_longa": justificativa_texto
    }
    return relatorio

# --- 4. FUNÇÃO: PREVISÃO EM LOTE ---

def predict_batch(jogos_do_dia, data_base):
    """Processa uma lista de jogos, gerando o relatório JSON para cada um."""
    relatorios = []
    
    for time_casa, time_fora in jogos_do_dia:
        relatorio = gerar_relatorio_json(time_casa, time_fora, data_base)
        relatorios.append(relatorio)
            
    return relatorios

# --- 5. FUNÇÃO: GERAÇÃO DE RANKING ---

def gerar_ranking_forca(data_base, num_times=10):
    """Calcula e ranqueia os times por 'Power Score' (Net xG) nas últimas 5 partidas."""
    if df_historico is None:
        return {"erro": "Histórico de dados não carregado."}

    data_ref = pd.to_datetime(data_base)
    df_filtered = df_historico[df_historico['match_date'] < data_ref]
    times_unicos = pd.concat([df_historico['home_team'], df_historico['away_team']]).unique()
    
    ranking_data = []
    
    for team in times_unicos:
        
        home_rows = df_filtered[df_filtered['home_team'] == team].sort_values('match_date', ascending=False)
        away_rows = df_filtered[df_filtered['away_team'] == team].sort_values('match_date', ascending=False)
        
        last_row = None
        prefix = None
        
        if not home_rows.empty and (away_rows.empty or home_rows.iloc[0]['match_date'] >= away_rows.iloc[0]['match_date']):
             last_row = home_rows.iloc[0]
             prefix = 'home_'
        elif not away_rows.empty:
             last_row = away_rows.iloc[0]
             prefix = 'away_'

        if last_row is not None and prefix is not None:
            try:
                xg_for_5 = last_row[f'{prefix}roll_xg_for_5']
                xg_against_5 = last_row[f'{prefix}roll_xg_against_5']
                goals_for_5 = last_row[f'{prefix}roll_goals_for_5']
                
                net_xg = xg_for_5 - xg_against_5
                performance_diff = goals_for_5 - xg_for_5

                ranking_data.append({
                    "time": team,
                    "net_xg_5": round(net_xg, 2),
                    "performance_diff_5": round(performance_diff, 2),
                    "xg_for_5": round(xg_for_5, 2)
                })
            except KeyError:
                continue
                
    if not ranking_data:
        return {"erro": "Nenhum time encontrado com histórico suficiente para ranking."}

    df_ranking = pd.DataFrame(ranking_data)
    
    df_ranking_forca = df_ranking.sort_values(by='net_xg_5', ascending=False).head(num_times).reset_index(drop=True)
    df_ranking_forca['rank'] = df_ranking_forca.index + 1
    
    df_ranking_valor = df_ranking.sort_values(by='performance_diff_5', ascending=False).head(num_times).reset_index(drop=True)
    df_ranking_valor['rank'] = df_ranking_valor.index + 1
    
    return {
        "data_referencia": data_base,
        "ranking_forca_net_xg": df_ranking_forca[['rank', 'time', 'net_xg_5', 'xg_for_5']].to_dict('records'),
        "ranking_valor_gols_xg": df_ranking_valor[['rank', 'time', 'performance_diff_5', 'net_xg_5']].to_dict('records')
    }

carregar_componentes()
