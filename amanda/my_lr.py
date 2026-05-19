import numpy as np

# =========================================================================
# 1. COMPONENTES MATEMÁTICOS ELEMENTARES (Seção 1.1.1 e 1.1.2)
# =========================================================================

def funcao_discriminante_linear(X, pesos, bias):
    """
    Calcula a combinação linear combinando os pesos e o bias para as entradas.
    Equação: z = beta_0 + beta_1*X_1 + ... + beta_n*X_n
    """
    return np.dot(X, pesos) + bias

def funcao_sigmoide(z):
    """
    Aplica a função logística para espremer qualquer valor real no intervalo [0, 1].
    Equação: g(z) = 1 / (1 + e^-z)
    """
    # np.clip evita estouro numérico (overflow) se z for muito grande/pequeno
    z = np.clip(z, -500, 500) 
    return 1 / (1 + np.exp(-z))

def calcular_log_loss(y_real, y_probabilidade):
    """
    Mapeia a entropia cruzada binária (métrica de erro global do modelo).
    Aplica uma penalidade logarítmica que cresce conforme a certeza errada.
    """
    N = len(y_real)
    # Evita log(0) adicionando um limiar infinitesimal (epsilon)
    eps = 1e-15
    y_probabilidade = np.clip(y_probabilidade, eps, 1 - eps)
    
    custo_individual = - (y_real * np.log(y_probabilidade) + (1 - y_real) * np.log(1 - y_probabilidade))
    return np.mean(custo_individual)

# =========================================================================
# 2. MECANISMO DE APRENDIZADO (Seção 1.1.2.2)
# =========================================================================

def calcular_gradientes(X, y_real, y_probabilidade):
    """
    Calcula o vetor de derivadas parciais da Log Loss em relação aos parâmetros.
    Indica a direção de máxima ascensão do erro.
    """
    N = len(y_real)
    erro_residual = y_probabilidade - y_real
    
    # Derivadas parciais conforme a dedução analítica
    gradiente_pesos = np.dot(X.T, erro_residual) / N
    gradiente_bias = np.sum(erro_residual) / N
    
    return gradiente_pesos, gradiente_bias

def atualizar_parametros(pesos, bias, grad_pesos, grad_bias, alpha):
    """
    Aplica a regra do Gradiente Descendente, movendo os coeficientes na
    direção oposta ao crescimento do erro.
    """
    novos_pesos = pesos - alpha * grad_pesos
    novo_bias = bias - alpha * grad_bias
    return novos_pesos, novo_bias

# =========================================================================
# 3. ROTINAS DE TREINAMENTO E PREDIÇÃO (Interface Principal)
# =========================================================================

def treinar_regressao_logistica(X, y, alpha=0.01, epocas=1000, tolerancia=1e-6):
    """
    Executa o loop iterativo de otimização até a convergência ou fim das épocas.
    """
    N_amostras, N_caracteristicas = X.shape
    
    # Inicialização dos coeficientes (chutes iniciais zerados/aleatórios)
    pesos = np.zeros(N_caracteristicas)
    bias = 0.0
    
    historico_custo = []
    
    for epoca in range(epocas):
        # 1. Forward Pass (Mapeamento Linear -> Sigmóide)
        z = funcao_discriminante_linear(X, pesos, bias)
        y_prob = funcao_sigmoide(z)
        
        # 2. Computar o Erro Atual
        custo = calcular_log_loss(y, y_prob)
        historico_custo.append(custo)
        
        # 3. Backward Pass (Cálculo das Inclinações)
        grad_pesos, grad_bias = calcular_gradientes(X, y, y_prob)
        
        # Critério de Parada Opcional: Se o gradiente for desprezível (convergência)
        if np.all(np.abs(grad_pesos) < tolerancia) and abs(grad_bias) < tolerancia:
            print(f" -> Modelo convergiu antecipadamente na época {epoca}")
            break
            
        # 4. Ajustar os Parâmetros
        pesos, bias = atualizar_parametros(pesos, bias, grad_pesos, grad_bias, alpha)
        
    return pesos, bias, historico_custo

def prever_probabilidade(X, pesos, bias):
    """Retorna as probabilidades contínuas no intervalo [0, 1] (Soft Classification)"""
    z = funcao_discriminante_linear(X, pesos, bias)
    return funcao_sigmoide(z)

def prever_classe(X, pesos, bias, limiar=0.5):
    """Discretiza a saída com base no Limiar de Decisão (Hard Classification)"""
    probabilidades = prever_probabilidade(X, pesos, bias)
    return np.where(probabilidades >= limiar, 1, 0)