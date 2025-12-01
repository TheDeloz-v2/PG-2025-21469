# PG-2025-21469
Proyecto de Graduación 2025 - Carnet: 21469


# Multimodal Financial Prediction Agent

Agente de Deep Learning multimodal para predicción del precio de Tesla, integrando datos de mercado, análisis de sentimiento de tweets y análisis técnico.

## 🚀 Instalación

```bash
git clone https://github.com/TheDeloz-v2/multimodal-financial-prediction-agent.git
cd multimodal-financial-prediction-agent
pip install -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

## 📊 Uso

### 1. Generar Datasets

```bash
python src/main.py
```

Genera 4 datasets procesados:
- **Market** (~1150 samples): Correlaciones con mercado
- **Sentiment** (~850 samples): Sentimiento de tweets de Musk
- **Stock** (~980 samples): Indicadores técnicos
- **Fusion** (~820 samples): Combinación de todos

### 2. Entrenar Modelos

**Entrenamiento simple (1 horizonte):**
```bash
python src/train_models.py
```

**Entrenamiento multi-horizonte (1d, 5d, 21d):**
```bash
python src/train_multi_horizon_models.py
```

### 3. Validar y Predecir

```bash
python src/run_trading_agent.py
```

## 🎯 Features

**Market Pipeline** (~42 features):
- Retornos rezagados de proveedores (NVDA, AMD, etc.)
- Tech peers (AAPL, MSFT, GOOGL)
- Competidores (F, GM, NIO)
- Índices (S&P 500, NASDAQ, VIX)
- PCA por grupo

**Sentiment Pipeline** (2 features):
- Sentiment score VADER [-1, 1]
- Engagement normalizado [0, 1]

**Stock Pipeline** (~26 features):
- Indicadores técnicos (RSI, MACD, Bollinger Bands)
- Decomposición temporal y wavelet
- Volatilidad y retornos

## 📁 Estructura

```
src/
├── main.py                          # Generar datasets
├── train_models.py                  # Entrenar modelos (1 horizonte)
├── train_multi_horizon_models.py    # Entrenar multi-horizonte
└── run_trading_agent.py             # Validar y predecir

data/
├── raw/                             # Datos originales
└── processed/fusion/                # Datasets procesados

models/                              # Modelos entrenados (.pth)
results/                             # Métricas y comparaciones
```

## 📈 Resultados

Modelos multi-horizonte con corrección de volatilidad:
- **Short (1d)**: ~56% accuracy direccional
- **Medium (5d)**: ~55% accuracy direccional  
- **Long (21d)**: ~54% accuracy direccional

## 🛠️ Tecnologías

- **PyTorch**: Deep learning
- **yfinance**: Datos financieros
- **NLTK/VADER**: Análisis de sentimiento
- **statsmodels**: Análisis de series temporales
- **scikit-learn**: Preprocessing y métricas

## 📝 Licencia

MIT License - Ver `LICENSE`

## 👤 Autor

Diego Lemus - 21469

**TheDeloz-v2** - [@TheDeloz-v2](https://github.com/TheDeloz-v2)
 
