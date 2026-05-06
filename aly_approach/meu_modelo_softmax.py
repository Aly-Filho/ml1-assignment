import numpy as np

def softmax(z):
    # Subtraímos o valor máximo de cada linha para estabilidade numérica (evita exp de números gigantes)
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def one_hot_encode(y, num_classes):
    # Converte labels (0, 1, 2...) em matriz binária
    return np.eye(num_classes)[y.astype(int)]

def cost_function(X, y_one_hot, W, b):
    m = X.shape[0]
    z = np.dot(X, W) + b
    probs = softmax(z)
    
    epsilon = 1e-7
    # Cross-Entropy Loss para múltiplas classes
    cost = -(1/m) * np.sum(y_one_hot * np.log(probs + epsilon))
    return cost

def gradient_function(X, y_one_hot, W, b):
    m = X.shape[0]
    z = np.dot(X, W) + b
    probs = softmax(z)
    
    # O erro é a diferença entre a probabilidade prevista e o valor real (one-hot)
    err = probs - y_one_hot
    
    grad_W = (1/m) * np.dot(X.T, err)
    grad_b = (1/m) * np.sum(err, axis=0)
    
    return grad_b, grad_W

def gradient_descent(X, y, alpha, iterations, num_classes):
    n = X.shape[1]
    # W agora é uma matriz (features x classes) e b é um vetor (classes)
    W = np.zeros((n, num_classes))
    b = np.zeros(num_classes)
    y_one_hot = one_hot_encode(y, num_classes)

    for i in range(iterations):
        grad_b, grad_W = gradient_function(X, y_one_hot, W, b)

        W = W - alpha * grad_W
        b = b - alpha * grad_b

        if i % 1000 == 0:
            print(f"Iteração {i}: Custo {cost_function(X, y_one_hot, W, b):.4f}")
    
    return W, b

def predict(X, W, b):
    z = np.dot(X, W) + b
    probs = softmax(z)
    # Retorna o índice da classe com maior probabilidade
    return np.argmax(probs, axis=1)

def executar_lr_softmax(X_train, X_test, y_train, y_test):
    """
    Função principal adaptada para Softmax.
    """
    # Conversão para NumPy
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    y_test_np = y_test.values if hasattr(y_test, 'values') else y_test

    # Determinar número de classes únicas
    classes_unicas = np.unique(y_train_np)
    num_classes = len(classes_unicas)
    
    # Mapear labels para 0 até num_classes-1 se necessário
    label_map = {val: i for i, val in enumerate(classes_unicas)}
    y_train_mapped = np.array([label_map[val] for val in y_train_np])
    y_test_mapped = np.array([label_map[val] for val in y_test_np])

    # Hiperparâmetros
    learning_rate = 0.1 # Softmax costuma convergir melhor com alpha ligeiramente maior
    iterations = 10000

    print(f"\nA iniciar Softmax Gradient Descent ({num_classes} classes)...")
    
    # Treino
    final_W, final_b = gradient_descent(X_train_np, y_train_mapped, learning_rate, iterations, num_classes)

    # Previsões
    preds_train = predict(X_train_np, final_W, final_b)
    preds_test = predict(X_test_np, final_W, final_b)
    
    # Acurácias
    acc_train = np.mean(preds_train == y_train_mapped) * 100
    acc_test = np.mean(preds_test == y_test_mapped) * 100

    print("\n" + "="*40)
    print("🎯 Resultados - Logistic Regression Softmax")
    print("="*40)
    print(f"Acurácia no Treino: {acc_train:.2f}%")
    print(f"Acurácia no Teste:  {acc_test:.2f}%")
    print("="*40)

    return preds_test