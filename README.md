# 🎯 K-means Clustering Pipeline - UI/UX Mejorada

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub](https://img.shields.io/github/stars/leiderdario/kmeans-clustering-pipeline?style=social)](https://github.com/leiderdario/kmeans-clustering-pipeline)

## 📋 Descripción

Pipeline completo de **análisis de clustering con K-means** con interfaz de usuario mejorada. Implementa todas las etapas del aprendizaje automático con visualizaciones profesionales y reportes HTML interactivos.

### ✨ Características Principales

1. ✅ **Selección y Carga de Datos** - CSV o DataFrame
2. 🔧 **Preparación Automática** - Imputación, codificación, normalización
3. 📊 **Análisis Exploratorio (EDA)** - Estadísticas, correlaciones, distribuciones
4. 🎯 **Reducción Dimensional (PCA)** - Simplificación inteligente
5. 🤖 **K-means Clustering** - Búsqueda automática de K óptimo
6. 🎨 **Visualizaciones** - Gráficos 2D/3D de alta calidad
7. 📄 **Reporte HTML** - Dashboard interactivo con resultados

### 🚀 **Nuevo: UI/UX Mejorada**

- ✅ Sin ventanas emergentes que bloquean la ejecución
- ✅ Mensajes con colores y emojis contextuales
- ✅ Organización automática de resultados
- ✅ Reporte HTML profesional
- ✅ Encoding UTF-8 para Windows

---

## 🚀 Instalación

### Requisitos previos
- Python 3.7 o superior
- pip (gestor de paquetes de Python)

### Instalación de dependencias

```bash
pip install -r requirements.txt
```

O instalar manualmente:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

---

## 💻 Uso del Script

### Opción 1: Ejecutar con datos de ejemplo

El script incluye un generador de datos sintéticos para demostración:

```bash
python kmeans_clustering.py
```

### Opción 2: Usar tus propios datos CSV

```python
from kmeans_clustering import run_kmeans_pipeline

# Pipeline automático completo
analyzer = run_kmeans_pipeline(filepath='tus_datos.csv')
```

### Opción 3: Control paso a paso (avanzado)

```python
from kmeans_clustering import KMeansAnalyzer

# Crear analizador
analyzer = KMeansAnalyzer(random_state=42)

# 1. Cargar datos
analyzer.load_data(filepath='tus_datos.csv')

# 2. Preparar datos (eliminar columnas innecesarias)
analyzer.prepare_data(drop_columns=['ID', 'Timestamp'])

# 3. Análisis exploratorio
analyzer.exploratory_analysis()

# 4. Reducción de dimensionalidad (opcional)
analyzer.reduce_dimensions(variance_threshold=0.95)

# 5. Encontrar K óptimo
analyzer.find_optimal_k(k_range=range(2, 10))

# 6. Entrenar modelo
analyzer.fit_kmeans(n_clusters=4, use_pca=True)

# 7. Visualizar clusters
analyzer.visualize_clusters(use_pca=True)

# 8. Analizar características
cluster_stats = analyzer.analyze_clusters()

# 9. Guardar resultados
analyzer.save_results('datos_clusterizados.csv')
```

---

## 📁 Estructura del Proyecto

```
K-means/database1/
│
├── kmeans_clustering.py      # Script principal
├── requirements.txt           # Dependencias
├── ejemplo_datos.csv          # Dataset de ejemplo
├── README.md                  # Este archivo
│
└── Salidas generadas:
    ├── correlation_matrix.png       # Matriz de correlación
    ├── distributions.png            # Distribuciones de variables
    ├── outliers_boxplot.png         # Detección de outliers
    ├── pca_variance.png             # Análisis de varianza PCA
    ├── optimal_k_analysis.png       # Búsqueda de K óptimo
    ├── clusters_2d.png              # Visualización 2D
    ├── clusters_3d.png              # Visualización 3D
    ├── cluster_features.png         # Características por cluster
    ├── cluster_analysis.csv         # Estadísticas de clusters
    └── clustered_data.csv           # Datos con etiquetas
```

---

## 🔍 Características Principales

### 1. Selección y Carga de Datos
- ✅ Carga desde CSV o DataFrame
- ✅ Validación automática de estructura
- ✅ Información detallada del dataset

### 2. Preparación de Datos
- 🔧 Manejo de valores faltantes (imputación automática)
- 🔧 Codificación de variables categóricas (Label Encoding)
- 🔧 Normalización (StandardScaler: μ=0, σ=1)
- 🔧 Identificación automática de tipos de variables

### 3. Análisis Exploratorio (EDA)
- 📊 Estadísticas descriptivas completas
- 📊 Matriz de correlación con heatmap
- 📊 Distribuciones de todas las variables
- 📊 Detección de outliers con boxplots
- 📊 Visualizaciones de alta calidad

### 4. Simplificación de Datos (PCA)
- 🎯 Reducción de dimensionalidad automática
- 🎯 Selección de componentes por varianza explicada
- 🎯 Visualización de varianza acumulada
- 🎯 Transformación reversible

### 5. Clustering K-means
- 🤖 Búsqueda automática de K óptimo
- 🤖 Método del codo automatizado
- 🤖 Múltiples métricas de evaluación:
  - Coeficiente de Silueta
  - Davies-Bouldin Index
  - Calinski-Harabasz Index
  - Inercia
- 🤖 Entrenamiento robusto con múltiples inicializaciones

### 6. Visualización de Resultados
- 🎨 Gráficos 2D y 3D de clusters
- 🎨 Visualización de centroides
- 🎨 Análisis de características por cluster
- 🎨 Todas las imágenes en alta resolución (300 DPI)

---

## 📊 Métricas de Evaluación

### Coeficiente de Silueta
- **Rango:** -1 a 1
- **Mejor:** Valores cercanos a 1
- **Interpretación:** Mide qué tan similar es un objeto a su propio cluster comparado con otros clusters

### Davies-Bouldin Index
- **Rango:** 0 a ∞
- **Mejor:** Valores más bajos
- **Interpretación:** Mide la separación entre clusters (valores bajos = mejor separación)

### Calinski-Harabasz Index
- **Rango:** 0 a ∞
- **Mejor:** Valores más altos
- **Interpretación:** Ratio de dispersión entre clusters vs dentro de clusters

### Inercia (Within-Cluster Sum of Squares)
- **Rango:** 0 a ∞
- **Mejor:** Valores más bajos
- **Interpretación:** Suma de distancias cuadradas de muestras a su centroide más cercano

---

## 🎓 Conceptos Clave

### ¿Qué es K-means?
K-means es un algoritmo de **aprendizaje no supervisado** que agrupa datos similares en K clusters. Cada observación pertenece al cluster con el centroide más cercano.

### Pasos del Algoritmo:
1. Inicializar K centroides aleatoriamente
2. Asignar cada punto al centroide más cercano
3. Recalcular centroides como el promedio de puntos asignados
4. Repetir pasos 2-3 hasta convergencia

### ¿Cuándo usar K-means?
- ✅ Segmentación de clientes
- ✅ Compresión de imágenes
- ✅ Detección de anomalías
- ✅ Análisis de patrones
- ✅ Agrupación de documentos

---

## 📝 Ejemplo Práctico

### Dataset de ejemplo: `ejemplo_datos.csv`

Contiene datos de clientes con las siguientes variables:
- **Age:** Edad del cliente
- **Income:** Ingreso anual
- **SpendingScore:** Puntuación de gasto (0-100)
- **MembershipYears:** Años de membresía
- **VisitsPerMonth:** Visitas mensuales
- **Category:** Categoría de cliente (Premium/Standard)
- **Region:** Región geográfica

### Resultado Esperado

El script automáticamente:
1. Identifica 4 clusters óptimos
2. Segmenta clientes por comportamiento
3. Genera visualizaciones explicativas
4. Proporciona estadísticas por cluster
5. Guarda resultados en archivos CSV y PNG

---

## ⚙️ Parámetros Personalizables

### KMeansAnalyzer()
```python
KMeansAnalyzer(random_state=42)  # Semilla para reproducibilidad
```

### prepare_data()
```python
prepare_data(
    target_column='etiqueta',      # Columna objetivo a excluir
    drop_columns=['ID', 'Fecha']   # Columnas a eliminar
)
```

### reduce_dimensions()
```python
reduce_dimensions(
    n_components=3,           # Número de componentes (None = automático)
    variance_threshold=0.95   # Varianza mínima a retener (95%)
)
```

### find_optimal_k()
```python
find_optimal_k(
    k_range=range(2, 11),    # Rango de K a evaluar
    use_pca=True             # Usar datos reducidos por PCA
)
```

### fit_kmeans()
```python
fit_kmeans(
    n_clusters=4,            # Número de clusters (None = automático)
    use_pca=True             # Usar datos reducidos
)
```

---

## 🛠️ Solución de Problemas

### Error: "No module named 'sklearn'"
```bash
pip install scikit-learn
```

### Error: "No se pueden crear gráficos"
- Verifica instalación de matplotlib
- En entornos sin GUI, usa: `matplotlib.use('Agg')`

### Advertencia: "Convergence warning"
- Normal en datasets complejos
- Aumenta `max_iter` en KMeans si persiste

### Dataset muy grande
- Activa `use_pca=True` para reducir dimensionalidad
- Considera muestreo aleatorio antes del análisis

---

## 📚 Referencias y Recursos

### Documentación Oficial
- [Scikit-learn K-means](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [PCA en Scikit-learn](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [Pandas](https://pandas.pydata.org/docs/)
- [Matplotlib](https://matplotlib.org/stable/contents.html)

### Artículos Académicos
- MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations"
- Arthur, D., & Vassilvitskii, S. (2007). "k-means++: The advantages of careful seeding"

---

## 👨‍💻 Autor

**Sistema de Análisis ML**  
Desarrollado para el SENA - Programa de Aprendizaje Automático  
Fecha: Octubre 2025

---

## 📄 Licencia

Este proyecto es de código abierto y está disponible para fines educativos.

---

## 🤝 Contribuciones

¿Encontraste un bug o tienes una sugerencia?
- Crea un issue con descripción detallada
- Fork el proyecto y envía pull requests
- Comparte tus casos de uso

---

## ⭐ Características Adicionales Futuras

- [ ] Soporte para clustering jerárquico
- [ ] Integración con DBSCAN y HDBSCAN
- [ ] Exportación a formatos interactivos (Plotly)
- [ ] Dashboard web con Streamlit
- [ ] API REST para predicciones
- [ ] Optimización con GPU (RAPIDS)

---

**¡Disfruta del análisis de clustering! 🚀📊**
