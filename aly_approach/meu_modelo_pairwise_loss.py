import numpy as np
from itertools import combinations

# ==========================================
# MOTOR MATEMÁTICO: FOCAL LOSS
# ==========================================

def sigmoid(z):
    # Limita z para evitar avisos de overflow no np.exp
    z = np.clip(z, -250, 250)
    return 1 / (1 + np.exp(-z))

def focal_cost_function(X, y, w, b, gamma=2.0):
    """
    Calcula o custo usando a Focal Loss em vez da Binary Cross-Entropy.
    """
    m = X.shape[0]
    z = np.dot(X, w) + b
    p = sigmoid(z)
    
    # Epsilon é vital para evitar log(0)
    epsilon = 1e-7
    
    # Cálculo vetorizado da Focal Loss para y=1 e y=0
    cost_1 = -y * ((1 - p) ** gamma) * np.log(p + epsilon)
    cost_0 = -(1 - y) * (p ** gamma) * np.log(1 - p + epsilon)
    
    cost = (1/m) * np.sum(cost_1 + cost_0)
    return cost

def focal_gradient_function(X, y, w, b, gamma=2.0):
    """
    Derivada rigorosa da Focal Loss em relação a z, propagada para os pesos.
    """
    m = X.shape[0]
    z = np.dot(X, w) + b
    p = sigmoid(z)
    epsilon = 1e-7
    
    # 1. Derivada dL/dz para amostras positivas (y = 1)
    grad_1 = ((1 - p) ** gamma) * (gamma * p * np.log(p + epsilon) - (1 - p))
    
    # 2. Derivada dL/dz para amostras negativas (y = 0)
    grad_0 = (p ** gamma) * (-gamma * (1 - p) * np.log(1 - p + epsilon) + p)
    
    # 3. Combinar os gradientes com base no rótulo real (y)
    err = y * grad_1 + (1 - y) * grad_0
    
    # 4. Atualização dos pesos (Álgebra linear padrão)
    grad_w = (1/m) * np.dot(X.T, err)
    grad_b = (1/m) * np.sum(err)
    
    return grad_b, grad_w

def gradient_descent(X, y, alpha, iterations, gamma=2.0, verbose=False):
    n = X.shape[1]
    w = np.zeros(n)
    b = 0

    for i in range(iterations):
        grad_b, grad_w = focal_gradient_function(X, y, w, b, gamma)
        
        w = w - alpha * grad_w
        b = b - alpha * grad_b

        # Apenas imprime de vez em quando se verbose for True
        if verbose and i % 1000 == 0:
            print(f"  Iteração {i}: Custo Focal {focal_cost_function(X, y, w, b, gamma):.4f}")
            
    return w, b

# ==========================================
# ARQUITETURA: PAIRWISE COUPLING (OvO)
# ==========================================

def treinar_pairwise(X, y, classes_unicas, alpha, iterations, gamma=2.0):
    """
    Treina K(K-1)/2 modelos (um para cada par de classes) usando Focal Loss.
    """
    modelos = {}
    pares = list(combinations(classes_unicas, 2))
    
    for i, j in pares:
        # Isolar dados do duelo específico
        mask = (y == i) | (y == j)
        X_par = X[mask]
        y_par_original = y[mask]
        
        # Mapear Classe 'i' para 1 e Classe 'j' para 0
        y_par = np.where(y_par_original == i, 1, 0)
        
        print(f"  -> A treinar duelo: Classe {i} vs Classe {j} (Gamma={gamma})...")
        # Treina o classificador binário passando o fator de foco
        w, b = gradient_descent(X_par, y_par, alpha, iterations, gamma=gamma, verbose=False)
        
        modelos[(i, j)] = (w, b)
        
    return modelos

def prever_pairwise(X, modelos, classes_unicas):
    """
    Acoplamento Aditivo (Soft Voting): Soma as probabilidades de todos os duelos.
    """
    m = X.shape[0]
    num_classes = len(classes_unicas)
    pontuacoes = np.zeros((m, num_classes))
    
    for (i, j), (w, b) in modelos.items():
        z = np.dot(X, w) + b
        prob_i = sigmoid(z)
        prob_j = 1 - prob_i 
        
        idx_i = np.where(classes_unicas == i)[0][0]
        idx_j = np.where(classes_unicas == j)[0][0]
        
        # Acoplamento contínuo das probabilidades
        pontuacoes[:, idx_i] += prob_i
        pontuacoes[:, idx_j] += prob_j
        
    # Vence a classe com maior soma de probabilidades
    indices_vencedores = np.argmax(pontuacoes, axis=1)
    return classes_unicas[indices_vencedores]

# ==========================================
# FUNÇÃO PRINCIPAL
# ==========================================

def executar_lr_pairwise_loss(X_train, X_test, y_train, y_test):
    """
    Função a ser chamada pelo ficheiro de teste.
    """
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    y_test_np = y_test.values if hasattr(y_test, 'values') else y_test

    classes_unicas = np.unique(y_train_np)
    num_classes = len(classes_unicas)
    num_modelos = int(num_classes * (num_classes - 1) / 2)

    # Hiperparâmetros
    learning_rate = 0.1
    iterations = 5000
    fator_foco = 2.0 # O teu super-poder matemático (Gamma)

    print(f"\nA iniciar Pairwise Coupling + Focal Loss ({num_classes} classes -> {num_modelos} modelos)...")
    
    # Treino
    modelos = treinar_pairwise(X_train_np, y_train_np, classes_unicas, learning_rate, iterations, gamma=fator_foco)

    # Previsões
    preds_train = prever_pairwise(X_train_np, modelos, classes_unicas)
    preds_test = prever_pairwise(X_test_np, modelos, classes_unicas)
    
    # Acurácias
    acc_train = np.mean(preds_train == y_train_np) * 100
    acc_test = np.mean(preds_test == y_test_np) * 100

    print("\n" + "="*50)
    print("🎯 Resultados - Logistic Regression (Pairwise + Focal)")
    print("="*50)
    print(f"Fator de Foco (Gamma): {fator_foco}")
    print(f"Modelos Treinados:     {num_modelos}")
    print(f"Acurácia no Treino:    {acc_train:.2f}%")
    print(f"Acurácia no Teste:     {acc_test:.2f}%")
    print("="*50)

    return preds_test