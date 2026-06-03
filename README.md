# Classificação de Áreas Propensas a Queimadas via Imagens Satelitais

Projeto de Applied Computer Vision desenvolvido para a Global Solution 2026 da FIAP, com foco no tema "Space Connect".

A solução usa redes neurais convolucionais treinadas do zero para classificar imagens de satélites em duas categorias: áreas com histórico de queimadas (`wildfire`) e áreas sem histórico (`nowildfire`). O objetivo é apoiar o monitoramento preventivo, identificando regiões com características associadas a queimadas históricas a partir de padrões de paisagem.

## Integrantes

- Guilherme Dal Posolo Matheus — RM98694
- Guilherme Faustino Vargas — RM98278
- João Lucas Yudi Redi Handa — RM98458
- Lucas Laia Manentti — RM97709
- Ryan Perez Pacheco — RM98782

## Resultados

| Modelo | Acurácia (teste) | AUC | Parâmetros |
| Baseline | 73,79% | 0,500 | 4.287.809 |
| Avançado | 97,30% | 0,997 | 423.265 |

## Estrutura do projeto

```
GlobalSolution/
├── app/app.py                        # Projeto Web do Streamlit
├── notebooks/                        # Notebooks do projeto
│   ├── 01_analise_exploratoria.ipynb
│   ├── 02_modelo_baseline.ipynb
│   ├── 03_modelo_avancado.ipynb
│   └── 04_avaliacao_comparacao.ipynb
├── models/                           # Pesos dos modelos treinados
├── reports/                          # Gráficos e métricas
├── amostras_teste/                   # Imagens para fazer testes rapdos
├── Dados/                            # Dataset
├── requirements.txt
└── README.md
```

## Como rodar o projeto

### 1. Criar e ativar o ambiente virtual

```bash
python3.12 -m venv venv
source venv/bin/activate
```

No Windows: `venv\Scripts\activate`

### 2. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 3. Rodar a aplicação Streamlit

```bash
streamlit run app/app.py
```

O navegador vai abrir automaticamente em `http://localhost:8501`.

## Como testar o projeto

A pasta `amostras_teste/` tem imagens prontas para fazer testes rapidos — 5 imagens de cada classe, extraídas do conjunto de teste (que o modelo nunca viu durante o treinamento).

Basta fazer o upload de uma imagem da pasta na aplicação web do Streamlit

## Reproduzir os experimentos

Os notebooks na pasta `notebooks/` tem que ser executados em ordem (01 → 04). Para refazer o treinamento, é preciso baixar o dataset completo:

```bash
kaggle datasets download -d abdelghaniaaba/wildfire-prediction-dataset -p Dados/
```

O dataset tem aproximadamente 42.850 imagens de sátelites de 350×350 pixels.

## Tecnologias

- Python 3.12
- TensorFlow 2.19.1 + tensorflow-metal (Apple Silicon)
- Streamlit
- scikit-learn, Matplotlib, Seaborn, Pandas, NumPy, Pillow