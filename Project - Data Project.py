import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import warnings
warnings.filterwarnings('ignore')

# =============================================================
# FASE 1 — Entendimento do Negócio
# =============================================================
# Problema de negócio:
# Investidores e analistas precisam avaliar o risco ESG de empresas do S&P 500.
# Classificar centenas de empresas manualmente é custoso e subjetivo.
#
# Objetivo analítico:
# Construir um modelo de Classificação que preveja o nível de risco ESG
# de uma empresa com base em seus scores ambientais, sociais e de governança.
#
# Variável alvo (y): `ESG Risk Level` - Low/Medium/High/Severe
# Critério de sucesso: Acurácia > 80% no conjunto de teste.
# Tipo de problema: Classificação Supervisionada (múltiplas classes)

CAMINHO_ARQUIVO = '/home/lucas/Documentos/Data Project/SP 500 ESG Risk Ratings.csv'

df = pd.read_csv(CAMINHO_ARQUIVO)

print(f"✅ Dataset carregado!")
print(f"   Linhas : {df.shape[0]}")
print(f"   Colunas: {df.shape[1]}")
print(f"\nColunas disponíveis:\n{list(df.columns)}")

print("=" * 60)
print("PRIMEIRAS LINHAS")
print("=" * 60)
print(df.head())

print("\n" + "=" * 60)
print("TIPOS E NULOS (.info)")
print("=" * 60)
df.info()

print("\n" + "=" * 60)
print("ESTATÍSTICAS DESCRITIVAS")
print("=" * 60)
print(df.describe())

print("=" * 60)
print("VALORES NULOS POR COLUNA")
print("=" * 60)

nulos = df.isnull().sum()
nulos_pct = (nulos / len(df) * 100).round(2)
tabela_nulos = pd.DataFrame({'Nulos': nulos, 'Percentual (%)': nulos_pct})
tabela_nulos = tabela_nulos[tabela_nulos['Nulos'] > 0]

if tabela_nulos.empty:
    print("✅ Nenhum valor nulo encontrado!")
else:
    print(tabela_nulos)

print("=" * 60)
print("DISTRIBUIÇÃO DA VARIÁVEL ALVO: ESG Risk Level")
print("=" * 60)
print(df['ESG Risk Level'].value_counts())

ordem = ['Low', 'Medium', 'High', 'Severe']
cores  = ['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad']
contagem = df['ESG Risk Level'].value_counts()
contagem = contagem.reindex([c for c in ordem if c in contagem.index])

plt.figure(figsize=(7, 4))
plt.bar(contagem.index, contagem.values,
        color=cores[:len(contagem)], edgecolor='white', width=0.6)
plt.title('Distribuição do ESG Risk Level', fontsize=13, fontweight='bold')
plt.xlabel('Nível de Risco ESG')
plt.ylabel('Quantidade de Empresas')
for i, v in enumerate(contagem.values):
    plt.text(i, v + 2, str(v), ha='center', fontsize=11)
plt.tight_layout()
plt.show()

colunas_score = [
    'Total ESG Risk Score',
    'Environment Risk Score',
    'Social Risk Score',
    'Governance Risk Score',
    'Controversy Score',
]
colunas_score = [c for c in colunas_score if c in df.columns]

fig, axes = plt.subplots(1, len(colunas_score),
                         figsize=(4 * len(colunas_score), 4))
if len(colunas_score) == 1:
    axes = [axes]

for ax, col in zip(axes, colunas_score):
    ax.hist(df[col].dropna(), bins=30,
            color='#3498db', edgecolor='white', alpha=0.85)
    ax.set_title(col, fontsize=9, fontweight='bold')
    ax.set_xlabel('Score')
    ax.set_ylabel('Frequência')

plt.suptitle('Distribuição dos Scores ESG',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Mapa de correlação
colunas_num_corr = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
colunas_num_corr = [c for c in colunas_num_corr if c != 'ESG Risk Percentile']

plt.figure(figsize=(9, 6))
corr = df[colunas_num_corr].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, square=True, cbar_kws={'shrink': 0.8})
plt.title('Correlação entre variáveis numéricas',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print("=" * 60)
print("FASE 3 — PREPARAÇÃO DOS DADOS")
print("=" * 60)

df_modelo = df.copy()

# Remover colunas que vazam informação do target ou são identificadores
colunas_remover = ['ESG Risk Percentile', 'Symbol', 'Name']
colunas_remover = [c for c in colunas_remover if c in df_modelo.columns]
df_modelo.drop(columns=colunas_remover, inplace=True)
print(f"✅ Removidas (leakage/identificadores): {colunas_remover}")

# Tratar nulos numéricos com mediana
colunas_num = df_modelo.select_dtypes(include=['float64', 'int64']).columns.tolist()

for col in colunas_num:
    qtd_nulos = df_modelo[col].isnull().sum()
    if qtd_nulos > 0:
        mediana = df_modelo[col].median()
        df_modelo[col].fillna(mediana, inplace=True)
        print(f"   '{col}': {qtd_nulos} nulos → preenchidos com mediana ({mediana:.2f})")

print("✅ Tratamento de nulos concluído!")

# Encoding das colunas categóricas (exceto o target)
colunas_cat = df_modelo.select_dtypes(include=['object']).columns.tolist()
colunas_cat = [c for c in colunas_cat if c != 'ESG Risk Level']
print(f"\n   Colunas para encoding: {colunas_cat}")

df_modelo = pd.get_dummies(df_modelo, columns=colunas_cat, drop_first=True)
print(f"✅ Encoding concluído! Dataset agora tem {df_modelo.shape[1]} colunas.")

# Remover linhas sem target
nulos_target = df_modelo['ESG Risk Level'].isnull().sum()
if nulos_target > 0:
    df_modelo.dropna(subset=['ESG Risk Level'], inplace=True)
    print(f"⚠️  {nulos_target} linhas sem ESG Risk Level removidas.")

# Separar X e y
X = df_modelo.drop(columns=['ESG Risk Level'])
y = df_modelo['ESG Risk Level']

print(f"\n✅ X: {X.shape[1]} features | {X.shape[0]} empresas")
print(f"✅ y: {y.nunique()} classes → {sorted(y.unique())}")

# Divisão treino / teste (70% / 30%)
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\n✅ Treino: {X_treino.shape[0]} empresas | Teste: {X_teste.shape[0]} empresas")

print("=" * 60)
print("FASE 4 — MODELAGEM")
print("=" * 60)

modelo = DecisionTreeClassifier(
    max_depth=4,
    class_weight='balanced',
    random_state=42
)

modelo.fit(X_treino, y_treino)
print("✅ Modelo treinado!")

y_pred = modelo.predict(X_teste)

print("=" * 60)
print("FASE 5 — AVALIAÇÃO DE DESEMPENHO")
print("=" * 60)

acuracia = accuracy_score(y_teste, y_pred)
print(f"\n🎯 Acurácia no conjunto de teste: {acuracia * 100:.2f}%")

if acuracia >= 0.80:
    print("   ✅ Critério de sucesso atingido (> 80%)!")
else:
    print("   ⚠️  Abaixo do critério de sucesso. Veja sugestões ao final.")

print("\n--- Relatório por classe ---")
print(classification_report(y_teste, y_pred))

# Matriz de confusão
fig, ax = plt.subplots(figsize=(7, 5))
cm   = confusion_matrix(y_teste, y_pred, labels=modelo.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=modelo.classes_)
disp.plot(ax=ax, colorbar=False, cmap='Blues')
plt.title('Matriz de Confusão — ESG Risk Level',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print("=" * 60)
print("TOP 10 FEATURES MAIS IMPORTANTES")
print("=" * 60)

importancias = pd.Series(modelo.feature_importances_, index=X.columns)
top10 = importancias.sort_values(ascending=False).head(10)
print(top10.reset_index().rename(columns={'index': 'Feature', 0: 'Importância'}))

plt.figure(figsize=(8, 5))
top10.sort_values().plot(kind='barh', color='#3498db', edgecolor='white')
plt.title('Top 10 Features — Importância para o Modelo',
          fontsize=13, fontweight='bold')
plt.xlabel('Importância')
plt.tight_layout()
plt.show()

print("=" * 60)
print("RESUMO DO PROJETO")
print("=" * 60)
print(f"  Dataset        : SP 500 ESG Risk Ratings.csv")
print(f"  Empresas       : {X.shape[0]}")
print(f"  Features usadas: {X.shape[1]}")
print(f"  Algoritmo      : DecisionTreeClassifier (max_depth=4)")
print(f"  Acurácia       : {acuracia * 100:.2f}%")
print()
print("Próximos passos sugeridos:")
print("  1. Testar max_depth=3 ou 5 e comparar acurácia")
print("  2. Remover features com importância = 0")
print("  3. Atividade extra: rodar sem 'Total ESG Risk Score' e comparar resultado")
print("  4. Para a banca: traduzir acurácia em impacto de negócio")
print("     Ex: 'O modelo identificou corretamente X% das empresas de alto risco'")
