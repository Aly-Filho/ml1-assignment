import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importando o algoritmo que construímos do zero!
import my_lr

# =====================================================================
# 1. FUNÇÕES DE MÉTRICAS E GRÁFICOS "FROM SCRATCH" (100% SEM SKLEARN)
# =====================================================================
def calcular_metricas(y_real, y_pred):
    """Calcula todas as métricas com base na Matriz de Confusão."""
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)
    
    TP = np.sum((y_real == 1) & (y_pred == 1))
    TN = np.sum((y_real == 0) & (y_pred == 0))
    FP = np.sum((y_real == 0) & (y_pred == 1))
    FN = np.sum((y_real == 1) & (y_pred == 0))
    
    eps = 1e-15 # Evita divisão por zero
    
    acuracia = (TP + TN) / len(y_real)
    precisao = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1_score = 2 * (precisao * recall) / (precisao + recall + eps)
    
    matriz_confusao = np.array([[TN, FP], [FN, TP]])
    
    return acuracia, precisao, recall, f1_score, matriz_confusao

def calcular_roc_auc(y_real, y_prob):
    """Calcula a Curva ROC e a Área AUC ordenando as probabilidades e usando a Regra do Trapézio."""
    y_real = np.array(y_real)
    y_prob = np.array(y_prob)
    
    # Ordena as probabilidades de forma decrescente
    indices_ordenados = np.argsort(y_prob)[::-1]
    y_real_ord = y_real[indices_ordenados]
    y_prob_ord = y_prob[indices_ordenados]
    
    Total_P = np.sum(y_real_ord == 1)
    Total_N = np.sum(y_real_ord == 0)
    
    if Total_P == 0 or Total_N == 0:
        return [0, 1], [0, 1], 0.5
        
    tpr_list, fpr_list = [0.0], [0.0]
    TP, FP = 0, 0
    
    for i in range(len(y_real_ord)):
        if y_real_ord[i] == 1:
            TP += 1
        else:
            FP += 1
            
        # Adiciona o ponto apenas quando a probabilidade muda (evita escadinhas irreais)
        if i == len(y_real_ord) - 1 or y_prob_ord[i] != y_prob_ord[i+1]:
            tpr_list.append(TP / Total_P)
            fpr_list.append(FP / Total_N)
            
    # Calcula a AUC pela área dos trapézios
    auc_score = 0.0
    for i in range(1, len(fpr_list)):
        largura = fpr_list[i] - fpr_list[i-1]
        altura_media = (tpr_list[i] + tpr_list[i-1]) / 2.0
        auc_score += largura * altura_media
        
    return fpr_list, tpr_list, auc_score

# =====================================================================
# 2. SISTEMA INTERATIVO DE AVALIAÇÃO EMPÍRICA
# =====================================================================
caminho_pasta = 'data_sets/class_imbalance/' 
arquivos_csv = sorted(glob.glob(os.path.join(caminho_pasta, "*.csv")))

if len(arquivos_csv) == 0:
    print("Nenhum arquivo encontrado. Verifique o caminho da pasta.")
else:
    # --- ETAPA A: ESCOLHA DO DATASET ---
    print(f"Encontramos {len(arquivos_csv)} datasets disponíveis.")
    escolha_ds = int(input(f"Escolha um dataset digitando um número de 1 a {len(arquivos_csv)}: ")) - 1
    
    arquivo_escolhido = arquivos_csv[escolha_ds]
    df = pd.read_csv(arquivo_escolhido)
    nome_ds = os.path.basename(arquivo_escolhido)
    
    col_alvo = df.columns[-1]
    total_linhas = len(df)
    qtd_features = len(df.columns) - 1
    contagem = df[col_alvo].value_counts()
    perc_min = (contagem.min() / total_linhas) * 100
    
    print("\n" + "="*50)
    print(f"📊 DATASET SELECIONADO: {nome_ds}")
    print(f"Linhas: {total_linhas} | Features: {qtd_features}")
    print(f"Desbalanceamento: A classe minoritária representa apenas {perc_min:.2f}% dos dados.")
    print("="*50 + "\n")
    
    # --- ETAPA B: PREPARAÇÃO DOS DADOS ---
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    X_bruto = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    classes = np.unique(y)
    y = np.where(y == classes[0], 0, 1)
    
    desvio = np.std(X_bruto, axis=0)
    desvio[desvio == 0] = 1e-15
    X = (X_bruto - np.mean(X_bruto, axis=0)) / desvio
    
    # --- ETAPA C: ESCOLHA DO MÉTODO DE SPLIT ---
    print("MÉTODOS DE VALIDAÇÃO:")
    print("1) Holdout (Divisão simples de Treino/Teste)")
    print("2) K-Fold Cross Validation")
    escolha_split = int(input("Escolha o método digitando 1 ou 2: "))
    
    if escolha_split == 1:
        pct_treino = float(input("Digite a porcentagem de treino (ex: 0.8 para 80%): "))
        limite = int(total_linhas * pct_treino)
        splits = [(X[:limite], y[:limite], X[limite:], y[limite:])]
        print(f"\nDividindo dados: {limite} para treino, {total_linhas - limite} para teste.")
        
    elif escolha_split == 2:
        k = int(input("Digite o número de Folds (K) (ex: 5): "))
        tamanho_fold = total_linhas // k
        splits = []
        print(f"\nIniciando validação cruzada com {k} Folds...")
        for i in range(k):
            inicio = i * tamanho_fold
            fim = inicio + tamanho_fold if i != k-1 else total_linhas
            X_ts_k, y_ts_k = X[inicio:fim], y[inicio:fim]
            X_tr_k = np.concatenate((X[:inicio], X[fim:]), axis=0)
            y_tr_k = np.concatenate((y[:inicio], y[fim:]), axis=0)
            splits.append((X_tr_k, y_tr_k, X_ts_k, y_ts_k))

    # --- ETAPA D: TREINAMENTO E AVALIAÇÃO ---
    todas_acuracias, todas_precisoes, todos_recalls, todos_f1s = [], [], [], []
    y_reais_completos, y_probs_completos = [], [] 
    matriz_confusao_soma = np.zeros((2,2))
    
    print("\n🚀 Treinando Regressão Logística Clássica (Padrão)...")
    
    for i, (X_tr, y_tr, X_ts, y_ts) in enumerate(splits):
        pesos, bias, historico = my_lr.treinar_regressao_logistica(X_tr, y_tr, alpha=0.1, epocas=1000)
        probs = my_lr.prever_probabilidade(X_ts, pesos, bias)
        preds = my_lr.prever_classe(X_ts, pesos, bias, limiar=0.5)
        
        y_reais_completos.extend(y_ts)
        y_probs_completos.extend(probs)
        
        acc, prec, rec, f1, matriz = calcular_metricas(y_ts, preds)
        todas_acuracias.append(acc)
        todas_precisoes.append(prec)
        todos_recalls.append(rec)
        todos_f1s.append(f1)
        matriz_confusao_soma += matriz

    acc_final = np.mean(todas_acuracias)
    prec_final = np.mean(todas_precisoes)
    rec_final = np.mean(todos_recalls)
    f1_final = np.mean(todos_f1s)
    
    fpr, tpr, auc_score = calcular_roc_auc(y_reais_completos, y_probs_completos)
    
    print("\n" + "="*50)
    print("📈 RESULTADOS DO BENCHMARKING (MODELO PADRÃO)")
    print("="*50)
    print(f"Acurácia Global : {acc_final * 100:.2f}%")
    print(f"Precisão        : {prec_final * 100:.2f}%")
    print(f"Sensibilidade (Recall): {rec_final * 100:.2f}%")
    print(f"F1-Score        : {f1_final * 100:.2f}%")
    print(f"AUC-ROC         : {auc_score:.4f}")
    
    # --- ETAPA E: VISUALIZAÇÕES E GRÁFICOS ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.heatmap(matriz_confusao_soma, annot=True, fmt='g', cmap='Blues', ax=ax1, 
                xticklabels=['Previsto 0', 'Previsto 1'], 
                yticklabels=['Real 0', 'Real 1'])
    ax1.set_title('Matriz de Confusão Agregada')
    
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {auc_score:.2f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Taxa de Falsos Positivos')
    ax2.set_ylabel('Taxa de Verdadeiros Positivos (Recall)')
    ax2.set_title('Receiver Operating Characteristic (ROC)')
    ax2.legend(loc="lower right")
    
    plt.tight_layout()
    plt.show()