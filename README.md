# Taller: Predicción de Retornos de Bitcoin con BiLSTM

**Maestría en Inteligencia de Negocios — Tópicos de ML y Redes Neuronales**  
**Autor:** Sebastian Angel  
**Fecha:** Abril 2026

---

## Contexto

Este taller es una extensión del notebook `05_btc_bilstm.ipynb` del libro *Redes Neuronales Recurrentes: Modelando Memoria y Secuencia*. El reto propuesto consistió en reemplazar la predicción de **precios absolutos** por la predicción de **retornos logarítmicos**, una práctica más sólida en el modelado de series financieras.

---

## Lo que se hizo

### 1. Configuración del entorno con Poetry

Se creó un ambiente virtual aislado usando **Poetry** como gestor de dependencias, apuntando específicamente a **Python 3.11.9** para garantizar compatibilidad con TensorFlow.

Dependencias instaladas:
- `numpy`, `pandas`, `matplotlib`
- `scikit-learn`
- `yfinance`
- `tensorflow`
- `jupyterlab`, `ipykernel`

El kernel fue registrado en Jupyter bajo el nombre `Python (btc_returns)` para mantener el ambiente aislado del sistema base.

---

### 2. Paso 1 — Serie de retornos logarítmicos

Se transformó la serie de precios de cierre diario de BTC-USD (2015–2026) a retornos log:

$$r_t = \log(P_t) - \log(P_{t-1})$$

**Observaciones:**
- La serie resultante oscila alrededor del **0 sin tendencia**, confirmando mayor estacionariedad frente al precio.
- La distribución es aproximadamente simétrica, centrada en 0, con **colas pesadas** (leptocurtosis).
- Se identificaron picos extremos alrededor de **2020** (crash de marzo COVID-19).
- Estadísticas clave: media ≈ 0.001, std ≈ 0.035, min ≈ -0.46, max ≈ 0.22.

---

### 3. Paso 2 — Ventanas temporales y split

Se construyó el dataset supervisado con ventanas de `LOOK_BACK = 60` días para predecir el retorno del día siguiente.

- Split temporal **90% / 10%** (sin mezcla de fechas para evitar *data leakage*).
- Escalado con `MinMaxScaler` ajustado **solo sobre el conjunto de entrenamiento**.

| Conjunto | Observaciones | Ventanas |
|----------|--------------|----------|
| Train    | 3,704        | 3,644    |
| Test     | 412          | 352      |

La diferencia de 60 entre observaciones y ventanas corresponde exactamente al `LOOK_BACK`.

---

### 4. Paso 3 — Arquitectura BiLSTM + Dropout

Se entrenó el mismo modelo del notebook original, ahora sobre retornos:

```
Bidirectional LSTM (50 unidades, return_sequences=True)
Dropout (0.2)
Bidirectional LSTM (50 unidades)
Dropout (0.2)
Dense (25, relu)
Dense (1)
```

- Total de parámetros: **83,751**
- Optimizador: `Adam`
- Loss: `MSE`
- Callbacks: `EarlyStopping (patience=5)` y `ReduceLROnPlateau (patience=3)`

El modelo entrenó **17 épocas** antes de detenerse. El learning rate se redujo automáticamente desde `0.001` hasta `6.25e-5`. No se observaron señales de overfitting severo.

---

### 5. Paso 4 — Reconstrucción del precio

A partir de los retornos predichos, se reconstruyó el precio acumulando:

$$P_t = P_{t-1} \cdot e^{r_t}$$

**Precio de arranque:** $96,125.55 USD (último día del conjunto train).

| Modelo | RMSE (USD) | MAE (USD) |
|--------|-----------|-----------|
| BiLSTM sobre retornos (este taller) | 16,460 | 13,911 |
| BiLSTM sobre precios (notebook original) | 3,458 | 2,629 |
| Baseline "mañana = hoy" | 2,071 | 1,499 |

El precio reconstruido derivó hacia abajo progresivamente, alejándose del precio real. Esto se debe al **error acumulado**: pequeños sesgos en cada retorno predicho se amplifican día a día.

---

### 6. Paso 5 — Accuracy direccional

En lugar de evaluar el error en USD (inadecuado para modelos de retornos), se midió cuántas veces el modelo acertó la **dirección** del movimiento (¿sube o baja?).

| Métrica | Resultado |
|---------|-----------|
| Accuracy direccional | **51.14%** |
| Baseline aleatorio | 50.00% |
| Días alcistas reales | 172 (48.9%) |
| Días bajistas reales | 180 (51.1%) |

**Matriz de confusión:**

|  | Real sube | Real baja |
|--|-----------|-----------|
| **Pred sube** | 43 | 43 |
| **Pred baja** | 129 | 137 |

El modelo mostró un **sesgo bajista marcado**: predijo "baja" el 74% de las veces, lo que explica la deriva descendente del precio reconstruido.

---

## Aprendizajes

### ¿Por qué trabajar con retornos?
- Los retornos son **más estacionarios** que los precios, lo que facilita el aprendizaje del modelo.
- Eliminan la tendencia creciente del precio, que puede llevar al modelo a simplemente "copiar el nivel anterior".
- Son la unidad estándar de análisis en finanzas cuantitativas.

### El problema del error acumulado
- Predecir retornos y reconstruir el precio acumula errores en cada paso.
- Un sesgo sistemático pequeño (predecir retornos levemente negativos) genera una divergencia enorme en el precio final.
- Por esto, **el RMSE en USD no es la métrica correcta** para evaluar un modelo de retornos.

### La métrica correcta es la dirección
- En finanzas, acertar la **dirección** del movimiento es más valioso que acertar el valor exacto.
- Un accuracy direccional **mayor al 55%** ya es considerado útil para estrategias de trading.
- El modelo alcanzó 51.14%, apenas por encima del azar, lo que indica que hay espacio de mejora significativo.

### El baseline "mañana = hoy" es difícil de superar
- En series financieras, el precio del día anterior es un predictor muy fuerte del precio actual.
- Los modelos de deep learning solo aportan valor real cuando capturan patrones no lineales que el baseline no puede.

---

## Posibles mejoras

- Agregar **features adicionales**: volumen, retornos de múltiples lags, indicadores técnicos (RSI, MACD, Bollinger Bands).
- Reformular como **clasificación binaria** (sube/baja) en lugar de regresión, optimizando directamente el accuracy direccional.
- Explorar **Attention mechanisms** o arquitecturas **Transformer** para series temporales.
- Ajustar el **umbral de decisión** (actualmente 0) para balancear las predicciones alcistas y bajistas.
- Usar **walk-forward validation** para una evaluación más robusta en el tiempo.

---

## Estructura del proyecto

```
taller_retornos_bitcoin/
  notebooks/
    06_btc_returns.ipynb
  pyproject.toml
  poetry.lock
```

---

*Generado como cierre del Reto Final del capítulo 5 del libro Redes Neuronales Recurrentes.*
