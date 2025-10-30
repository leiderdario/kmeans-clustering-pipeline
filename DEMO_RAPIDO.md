# 🚀 Demo Rápido - K-means Clustering UI/UX Mejorada

## Ejecutar el Script

### 1️⃣ Con datos de ejemplo (automático)
```bash
python kmeans_clustering.py
```

### 2️⃣ Con tu propio archivo CSV
```python
from kmeans_clustering import run_kmeans_pipeline

analyzer = run_kmeans_pipeline(filepath='ejemplo_datos.csv')
```

### 3️⃣ Personalizado
```python
from kmeans_clustering import KMeansAnalyzer

# Crear analizador con directorio personalizado
analyzer = KMeansAnalyzer(output_dir='mi_analisis')

# Pipeline paso a paso
analyzer.load_data(filepath='ejemplo_datos.csv')
analyzer.prepare_data()
analyzer.exploratory_analysis()
analyzer.reduce_dimensions(variance_threshold=0.90)
analyzer.find_optimal_k(k_range=range(2, 8))
analyzer.fit_kmeans(n_clusters=4)
analyzer.visualize_clusters()
analyzer.analyze_clusters()
analyzer.save_results('resultados_finales.csv')
analyzer.generate_report()
```

---

## ✅ Resultados Generados

Después de ejecutar, encontrarás en `resultados_kmeans/`:

| Archivo | Descripción |
|---------|-------------|
| `reporte_kmeans.html` | 📊 Reporte interactivo (¡ABRE ESTE!) |
| `correlation_matrix.png` | Matriz de correlación |
| `distributions.png` | Distribuciones de variables |
| `outliers_boxplot.png` | Detección de outliers |
| `pca_variance.png` | Análisis PCA |
| `optimal_k_analysis.png` | Búsqueda de K óptimo |
| `clusters_2d.png` | Visualización 2D |
| `clusters_3d.png` | Visualización 3D |
| `cluster_features.png` | Características por cluster |
| `cluster_analysis.csv` | Estadísticas por cluster |
| `clustered_data.csv` | Datos con etiquetas |

---

## 🎯 Ejemplo Práctico

```python
# ejemplo_uso.py
from kmeans_clustering import run_kmeans_pipeline

# Ejecutar análisis completo
analyzer = run_kmeans_pipeline(
    filepath='ejemplo_datos.csv',
    n_clusters=None,        # Búsqueda automática
    use_pca=True,           # Reducir dimensionalidad
    generate_html=True      # Generar reporte HTML
)

# Acceder a resultados
print(f"K óptimo encontrado: {analyzer.optimal_k}")
print(f"Archivos generados: {len(analyzer.plots_generated)}")

# Ver centroides
print("\nCentroides de los clusters:")
print(analyzer.kmeans.cluster_centers_)

# Distribución de clusters
import pandas as pd
print("\nDistribución:")
print(pd.Series(analyzer.labels).value_counts().sort_index())
```

---

## 🎨 Características UI/UX

### Antes
```
📊 ANÁLISIS EXPLORATORIO DE DATOS
============================================================
[ventana emergente bloquea ejecución]
```

### Ahora
```
======================================================================
                    📊 ANÁLISIS EXPLORATORIO (EDA)
======================================================================
ℹ️  Estadísticas Descriptivas:
[tabla de datos]
   📊 Filas: 300
   📊 Columnas: 7
   📊 Memoria: 0.04 MB
ℹ️  Generando matriz de correlación...
✅ Guardado: correlation_matrix.png
ℹ️  Generando distribuciones de variables...
✅ Guardado: distributions.png
✅ Análisis exploratorio completado
```

---

## 💡 Tips

1. **Ver reporte HTML:** Abre `resultados_kmeans/reporte_kmeans.html` en tu navegador
2. **Cambiar directorio:** `KMeansAnalyzer(output_dir='otro_nombre')`
3. **Sin PCA:** `run_kmeans_pipeline(use_pca=False)`
4. **K fijo:** `run_kmeans_pipeline(n_clusters=5)`
5. **Sin reporte:** `run_kmeans_pipeline(generate_html=False)`

---

## 🐛 Solución de Problemas

### Error: No module named 'tqdm'
```bash
pip install tqdm colorama
```
*Nota: Son opcionales, el script funciona sin ellas*

### Error de encoding en Windows
✅ Ya está solucionado con configuración automática UTF-8

### Los gráficos no se ven
✅ Ahora se guardan automáticamente en archivos PNG

---

## 📚 Documentación Completa

- `README.md` - Documentación principal
- `MEJORAS_UI_UX.md` - Detalles de las mejoras
- `requirements.txt` - Dependencias

---

**¡Listo para usar! 🎉**
