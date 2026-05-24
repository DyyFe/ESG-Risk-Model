# S&P 500 ESG Risk Classification

Projeto de classificação de risco ESG para empresas do S&P 500 usando Decision Tree. Feito como exercício de Machine Learning supervisionado.

---

## O problema

Avaliar o risco ESG de centenas de empresas manualmente é inviável — demora, custa caro e ainda fica sujeito à interpretação de quem faz. A ideia aqui foi treinar um modelo que faça essa classificação automaticamente a partir dos scores já disponíveis.

A variável alvo é o `ESG Risk Level`, que pode ser: **Low, Medium, High ou Severe**.

Meta: acurácia acima de 80% no conjunto de teste.

---

## Dataset

Dados públicos do Kaggle: [S&P 500 ESG Risk Ratings](https://www.kaggle.com/datasets/pritish509/s-and-p-500-esg-risk-ratings/data)

As features principais que o modelo usa são os scores de risco:

- `Total ESG Risk Score`
- `Environment Risk Score`
- `Social Risk Score`
- `Governance Risk Score`
- `Controversy Score`

> `ESG Risk Percentile` foi removido porque vaza a informação do target. `Symbol` e `Name` também saíram por serem só identificadores.

---

## Como rodar

Instala as dependências:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Baixa o dataset no Kaggle, salva em algum lugar e ajusta essa linha no script:

```python
CAMINHO_ARQUIVO = '/seu/caminho/SP 500 ESG Risk Ratings.csv'
```

Depois é só rodar:

```bash
python esg_classification.py
```

---

## O que o script faz

1. Carrega e explora os dados (distribuições, correlações, nulos)
2. Trata os nulos com mediana e faz encoding das categóricas
3. Divide em treino (70%) e teste (30%), estratificado por classe
4. Treina uma `DecisionTreeClassifier` com `max_depth=4`
5. Avalia com acurácia, relatório por classe e matriz de confusão
6. Plota as top 10 features mais importantes

---

## Resultado

A acurácia aparece no output ao rodar. Se ficar abaixo de 80%, vale testar `max_depth=3` ou `max_depth=5` e ver o que muda.

Uma coisa interessante pra testar também: rodar o modelo sem o `Total ESG Risk Score` e comparar — dá pra ver o quanto o modelo depende dessa feature específica.

---

## Fonte dos dados

https://www.kaggle.com/datasets/pritish509/s-and-p-500-esg-risk-ratings/data
