import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Adicionamos estas importações para as métricas
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

try:
    from meu_modelo_lr import executar_lr_original
    from meu_modelo_softmax import executar_lr_softmax
    modulos_carregados = True
except ImportError:
    modulos_carregados = False
    print("⚠️ Aviso: Os ficheiros dos teus modelos ainda não foram encontrados.")

def selecionar_dataset():
    diretorio_deste_script = os.path.dirname(os.path.abspath(__file__))
    caminho_pasta = os.path.join(diretorio_deste_script, "..", "data_sets", "data_sets_reworked")
    caminho_pasta = os.path.abspath(caminho_pasta)
    
    if not os.path.exists(caminho_pasta):
        print(f"Erro: A pasta não foi encontrada no caminho: {caminho_pasta}")
        return None
        
    arquivos = [f for f in os.listdir(caminho_pasta) if f.endswith('.csv')]
    
    if not arquivos:
        print("Nenhum ficheiro .csv encontrado na pasta.")
        return None
        
    print("="*40)
    print("Qual data set você quer testar?")
    print("="*40)
    for i, arquivo in enumerate(arquivos):
        print(f"[{i + 1}] {arquivo}")
    print("="*40)
    
    while True:
        escolha = input("\nDigite o número correspondente ao dataset: ")
        try:
            indice = int(escolha) - 1
            if 0 <= indice < len(arquivos):
                nome_dataset = arquivos[indice]
                caminho_completo = os.path.join(caminho_pasta, nome_dataset)
                print(f"\n✅ Dataset '{nome_dataset}' selecionado com sucesso!")
                return caminho_completo
            else:
                print("⚠️ Número fora da lista. Tente novamente.")
        except ValueError:
            print("⚠️ Por favor, digite apenas números inteiros.")

def preparar_dados(caminho):
    df = pd.read_csv(caminho)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("\n✅ Dataset separado em Treino (80%) e Teste (20%).")
    
    colunas_numericas = X_train.select_dtypes(include=['float64', 'int64']).columns
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    if len(colunas_numericas) > 0:
        scaler = StandardScaler()
        X_train_scaled[colunas_numericas] = scaler.fit_transform(X_train[colunas_numericas])
        X_test_scaled[colunas_numericas] = scaler.transform(X_test[colunas_numericas])
        print(f"✅ Standardização aplicada.")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def mostrar_metricas(y_true, y_pred, nome_modelo):
    """
    Função auxiliar para imprimir um relatório bonito.
    """
    print("\n" + "="*50)
    print(f"📊 RELATÓRIO DE DESEMPENHO: {nome_modelo}")
    print("="*50)
    print(f"Acurácia Global: {accuracy_score(y_true, y_pred):.4f}")
    print("\nRelatório por Classe:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Matriz de Confusão:")
    print(confusion_matrix(y_true, y_pred))
    print("="*50)

def executar_modelos(X_train, X_test, y_train, y_test):
    print("\n" + "="*40)
    print("Selecione a abordagem:")
    print("[1] Logistic Regression (Original)")
    print("[2] Logistic Regression Soft-Max")
    print("[3] Ambos")
    print("="*40)
    
    while True:
        escolha = input("\nDigite a sua escolha (1, 2 ou 3): ")
        if escolha in ['1', '2', '3']:
            break
        print("⚠️ Opção inválida.")
        
    if not modulos_carregados:
        print("\n❌ Ficheiros externos não encontrados.")
        return

    # 1. Regressão Logística Original
    if escolha in ['1', '3']:
        print("\n⚙️ A treinar Regressão Logística Original...")
        # Capturamos o que a função retorna (as previsões do set de teste)
        preds = executar_lr_original(X_train, X_test, y_train, y_test)
        if preds is not None:
            mostrar_metricas(y_test, preds, "Logistic Regression (Original)")

    # 2. Regressão Logística Soft-Max
    if escolha in ['2', '3']:
        print("\n⚙️ A treinar Regressão Logística Soft-Max...")
        # Capturamos o que a função retorna
        preds_sm = executar_lr_softmax(X_train, X_test, y_train, y_test)
        if preds_sm is not None:
            mostrar_metricas(y_test, preds_sm, "Soft-Max Regression")

if __name__ == "__main__":
    caminho = selecionar_dataset()
    if caminho:
        X_train, X_test, y_train, y_test = preparar_dados(caminho)
        executar_modelos(X_train, X_test, y_train, y_test)