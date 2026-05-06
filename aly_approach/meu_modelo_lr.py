import numpy as np

def sigmoid(z):
    # np.clip limita os valores de z para evitar overflow de exponenciais
    z = np.clip(z, -250, 250)
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, w, b):
    m = X.shape[0]
    z = np.dot(X, w) + b
    g = sigmoid(z)
    
    # epsilon evita erro matemático de log(0)
    epsilon = 1e-7
    cost = -(1/m) * np.sum(y * np.log(g + epsilon) + (1 - y) * np.log(1 - g + epsilon))
    
    return cost

def gradient_function(X, y, w, b):
    m = X.shape[0]
    
    # Vectorização: calcula todas as previsões e erros de uma só vez
    z = np.dot(X, w) + b
    g = sigmoid(z)
    err = g - y
    
    # Produto escalar transpõe X e multiplica pelos erros para calcular todos os gradientes de w
    grad_w = (1/m) * np.dot(X.T, err)
    grad_b = (1/m) * np.sum(err)
    
    return grad_b, grad_w

def gradient_descent(X, y, alpha, iterations):
    n = X.shape[1]
    w = np.zeros(n)
    b = 0

    for i in range(iterations):
        grad_b, grad_w = gradient_function(X, y, w, b)

        w = w - alpha * grad_w
        b = b - alpha * grad_b

        # Print a cada 1000 iterações para acompanhar a convergência
        if i % 1000 == 0:
            print(f"Iteração {i}: Custo {cost_function(X, y, w, b):.4f}")
    
    return w, b

def predict(X, w, b):
    z = np.dot(X, w) + b
    g = sigmoid(z)
    
    # Retorna 1 se g >= 0.5, caso contrário 0 (forma vectorizada)
    return (g >= 0.5).astype(int)

def executar_lr_original(X_train, X_test, y_train, y_test):
    """
    Função principal que será chamada pelo test.py.
    Recebe os dados já separados e padronizados.
    """
    
    # O test.py envia os dados como DataFrames do Pandas, 
    # por isso convertemos para NumPy arrays para os cálculos de álgebra linear:
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    y_test_np = y_test.values if hasattr(y_test, 'values') else y_test

    # Hiperparâmetros
    learning_rate = 0.01
    iterations = 10000

    print(f"\nA iniciar Gradient Descent com alpha={learning_rate} e {iterations} iterações...")
    
    # Treino
    final_w, final_b = gradient_descent(X_train_np, y_train_np, learning_rate, iterations)

    # Previsões
    preds_train = predict(X_train_np, final_w, final_b)
    preds_test = predict(X_test_np, final_w, final_b)
    
    # Acurácias
    acc_train = np.mean(preds_train == y_train_np) * 100
    acc_test = np.mean(preds_test == y_test_np) * 100

    # Outputs Finais
    print("\n" + "="*40)
    print("🎯 Resultados - Logistic Regression (From Scratch)")
    print("="*40)
    print(f"Acurácia no Treino: {acc_train:.2f}%")
    print(f"Acurácia no Teste:  {acc_test:.2f}%")
    print("="*40)
    
    return preds_test