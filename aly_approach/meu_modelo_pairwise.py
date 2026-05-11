import numpy as np
from itertools import combinations

# ==========================================
# MATEMÁTICA BASE DA REGRESSÃO LOGÍSTICA
# ==========================================

def sigmoid(z):
    z = np.clip(z, -250, 250)
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, w, b):
    m = X.shape[0]
    z = np.dot(X, w) + b
    g = sigmoid(z)
    
    epsilon = 1e-7
    cost = -(1/m) * np.sum(y * np.log(g + epsilon) + (1 - y) * np.log(1 - g + epsilon))
    return cost

def gradient_function(X, y, w, b):
    m = X.shape[0]
    z = np.dot(X, w) + b
    g = sigmoid(z)
    err = g - y
    
    grad_w = (1/m) * np.dot(X.T, err)
    grad_b = (1/m) * np.sum(err)
    return grad_b, grad_w

def gradient_descent(X, y, alpha, iterations, verbose=False):
    n = X.shape[1]
    w = np.zeros(n)
    b = 0

    for i in range(iterations):
        grad_b, grad_w = gradient_function(X, y, w, b)
        w = w - alpha * grad_w
        b = b - alpha * grad_b

        # Apenas fazemos print se verbose=True para não poluir o terminal, 
        # já que vamos treinar vários modelos!
        if verbose and i % 1000 == 0:
            print(f"  Iteração {i}: Custo {cost_function(X, y, w, b):.4f}")
            
    return w, b

# ==========================================
# ARQUITETURA PAIRWISE COUPLING (ONE-VS-ONE)
# ==========================================

def treinar_pairwise(X, y, classes_unicas, alpha, iterations):
    """
    Treina K(K-1)/2 modelos, um para cada par de classes.
    """
    modelos = {}
    # Cria todas as combinações possíveis de pares (ex: (0,1), (0,2), (1,2))
    pares = list(combinations(classes_unicas, 2))
    
    for i, j in pares:
        # 1. Isolar apenas os dados que pertencem à classe i ou à classe j
        mask = (y == i) | (y == j)
        X_par = X[mask]
        y_par_original = y[mask]
        
        # 2. Binarizar as classes para este modelo específico:
        # A classe 'i' torna-se 1 (Positiva) e a classe 'j' torna-se 0 (Negativa)
        y_par = np.where(y_par_original == i, 1, 0)
        
        print(f"  -> A treinar duelo: Classe {i} vs Classe {j} ...")
        w, b = gradient_descent(X_par, y_par, alpha, iterations, verbose=False)
        
        # Guardar os pesos deste modelo no dicionário
        modelos[(i, j)] = (w, b)
        
    return modelos

def prever_pairwise(X, modelos, classes_unicas):
    """
    Acopla as probabilidades de todos os modelos para gerar a previsão final.
    """
    m = X.shape[0]
    num_classes = len(classes_unicas)
    
    # Matriz para acumular as probabilidades de cada classe para cada amostra
    pontuacoes = np.zeros((m, num_classes))
    
    # Avaliar a amostra em TODOS os modelos treinados
    for (i, j), (w, b) in modelos.items():
        z = np.dot(X, w) + b
        
        # Probabilidade de ser a classe i (pois mapeámos i para 1 no treino)
        prob_i = sigmoid(z)
        # Probabilidade de ser a classe j (o inverso da classe i)
        prob_j = 1 - prob_i 
        
        # Descobrir em que coluna da matriz de pontuações devemos somar
        idx_i = np.where(classes_unicas == i)[0][0]
        idx_j = np.where(classes_unicas == j)[0][0]
        
        # ACOPLAMENTO: Somamos as probabilidades contínuas (Soft Voting)
        pontuacoes[:, idx_i] += prob_i
        pontuacoes[:, idx_j] += prob_j
        
    # A classe vencedora é aquela que acumulou a maior soma de probabilidades
    indices_vencedores = np.argmax(pontuacoes, axis=1)
    
    # Mapear de volta para o nome original das classes
    return classes_unicas[indices_vencedores]

# ==========================================
# FUNÇÃO PRINCIPAL DE EXECUÇÃO
# ==========================================

def executar_lr_pairwise(X_train, X_test, y_train, y_test):
    # Conversão para NumPy
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    y_test_np = y_test.values if hasattr(y_test, 'values') else y_test

    classes_unicas = np.unique(y_train_np)
    num_classes = len(classes_unicas)
    num_modelos = int(num_classes * (num_classes - 1) / 2)

    # Hiperparâmetros
    learning_rate = 0.1
    iterations = 5000 # Reduzimos as iterações porque treinar múltiplos modelos demora mais

    print(f"\nA iniciar Pairwise Coupling ({num_classes} classes -> {num_modelos} modelos)...")
    
    # Treino
    modelos = treinar_pairwise(X_train_np, y_train_np, classes_unicas, learning_rate, iterations)

    # Previsões
    preds_train = prever_pairwise(X_train_np, modelos, classes_unicas)
    preds_test = prever_pairwise(X_test_np, modelos, classes_unicas)
    
    # Acurácias
    acc_train = np.mean(preds_train == y_train_np) * 100
    acc_test = np.mean(preds_test == y_test_np) * 100

    print("\n" + "="*45)
    print("🎯 Resultados - Logistic Regression (Pairwise)")
    print("="*45)
    print(f"Modelos treinados independentemente: {num_modelos}")
    print(f"Acurácia no Treino: {acc_train:.2f}%")
    print(f"Acurácia no Teste:  {acc_test:.2f}%")
    print("="*45)

    return preds_test