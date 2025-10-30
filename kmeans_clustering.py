

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Modo no-interactivo (sin ventanas emergentes)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.impute import SimpleImputer
import warnings
import os
import sys
from datetime import datetime
from pathlib import Path

# Configurar encoding para Windows (soluciona problemas con emojis)
if sys.platform == 'win32':
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except:
        pass

warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Intentar importar librerías para UI mejorada
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("💡 Tip: Instala 'tqdm' para barras de progreso: pip install tqdm")

try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    # Definir clases dummy si no está disponible
    class Fore:
        GREEN = CYAN = YELLOW = RED = MAGENTA = BLUE = WHITE = RESET = ''
    class Style:
        BRIGHT = RESET_ALL = ''


# ============================================================================
# FUNCIONES DE UI MEJORADAS
# ============================================================================

def print_header(text, color=Fore.CYAN):
    """Imprime un encabezado con formato mejorado."""
    if COLORAMA_AVAILABLE:
        print(f"\n{color}{Style.BRIGHT}{'='*70}")
        print(f"{text.center(70)}")
        print(f"{'='*70}{Style.RESET_ALL}")
    else:
        print(f"\n{'='*70}")
        print(f"{text.center(70)}")
        print(f"{'='*70}")

def print_success(text):
    """Imprime mensaje de éxito."""
    symbol = "✅"
    if COLORAMA_AVAILABLE:
        print(f"{Fore.GREEN}{symbol} {text}{Style.RESET_ALL}")
    else:
        print(f"{symbol} {text}")

def print_info(text):
    """Imprime mensaje informativo."""
    symbol = "ℹ️"
    if COLORAMA_AVAILABLE:
        print(f"{Fore.CYAN}{symbol}  {text}{Style.RESET_ALL}")
    else:
        print(f"{symbol}  {text}")

def print_warning(text):
    """Imprime mensaje de advertencia."""
    symbol = "⚠️"
    if COLORAMA_AVAILABLE:
        print(f"{Fore.YELLOW}{symbol}  {text}{Style.RESET_ALL}")
    else:
        print(f"{symbol}  {text}")

def print_error(text):
    """Imprime mensaje de error."""
    symbol = "❌"
    if COLORAMA_AVAILABLE:
        print(f"{Fore.RED}{symbol} {text}{Style.RESET_ALL}")
    else:
        print(f"{symbol} {text}")

def print_metric(label, value, unit=""):
    """Imprime una métrica con formato."""
    if COLORAMA_AVAILABLE:
        print(f"{Fore.MAGENTA}   📊 {label}:{Style.RESET_ALL} {Fore.WHITE}{value}{unit}{Style.RESET_ALL}")
    else:
        print(f"   📊 {label}: {value}{unit}")

def progress_bar(iterable, desc="Procesando"):
    """Crea una barra de progreso si tqdm está disponible."""
    if TQDM_AVAILABLE:
        return tqdm(iterable, desc=desc, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    else:
        return iterable


class KMeansAnalyzer:
    """
    Clase principal para realizar análisis de clustering con K-means.
    Implementa todo el pipeline de ML: carga, preparación, EDA, simplificación y clustering.
    """
    
    def __init__(self, random_state=42, output_dir='results'):
        """
        Inicializa el analizador de K-means.
        
        Args:
            random_state (int): Semilla para reproducibilidad
            output_dir (str): Directorio para guardar resultados
        """
        self.random_state = random_state
        self.df = None
        self.df_processed = None
        self.scaler = StandardScaler()
        self.pca = None
        self.kmeans = None
        self.optimal_k = None
        self.numeric_features = None
        self.categorical_features = None
        self.output_dir = output_dir
        self.plots_generated = []
        self.metrics = {}  # Almacenar métricas de evaluación
        self.execution_time = None
        
        # Crear directorio de salida
        Path(output_dir).mkdir(exist_ok=True)
        print_success(f"Carpeta '{output_dir}/' creada o actualizada")
    
    def _save_plot(self, filename, dpi=300):
        """Guarda un gráfico y cierra la figura."""
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        self.plots_generated.append(filepath)
        print_success(f"Guardado: {filename}")
        plt.close()  # Cerrar figura para liberar memoria
        
    # ============================================================================
    # 1. SELECCIÓN Y CARGA DE DATOS
    # ============================================================================
    
    def load_data(self, filepath=None, data=None):
        """
        Carga datos desde un archivo CSV o acepta un DataFrame.
        
        Args:
            filepath (str): Ruta al archivo CSV
            data (DataFrame): DataFrame de pandas
        """
        print_header("📂 CARGANDO DATOS", Fore.CYAN)
        
        if filepath:
            print_info(f"Origen: {filepath}")
            self.df = pd.read_csv(filepath)
        elif data is not None:
            print_info("Origen: DataFrame en memoria")
            self.df = data.copy()
        else:
            print_error("Debe proporcionar filepath o data")
            raise ValueError("Debe proporcionar filepath o data")
        
        print_success(f"Datos cargados: {self.df.shape[0]} filas × {self.df.shape[1]} columnas")
        print_info(f"Columnas: {', '.join(list(self.df.columns)[:5])}{'...' if len(self.df.columns) > 5 else ''}")
        return self
    
    # ============================================================================
    # 2. PREPARACIÓN DE DATOS
    # ============================================================================
    
    def prepare_data(self, target_column=None, drop_columns=None):
        """
        Prepara los datos: manejo de valores faltantes, codificación y normalización.
        
        Args:
            target_column (str): Columna objetivo a excluir del análisis
            drop_columns (list): Lista de columnas a eliminar
        """
        print_header("🔧 PREPARACIÓN DE DATOS", Fore.YELLOW)
        
        df_work = self.df.copy()
        
        # Eliminar columnas especificadas
        if drop_columns:
            df_work = df_work.drop(columns=drop_columns, errors='ignore')
            print_info(f"Columnas eliminadas: {', '.join(drop_columns)}")
        
        # Separar variable objetivo si existe
        if target_column and target_column in df_work.columns:
            self.target = df_work[target_column]
            df_work = df_work.drop(columns=[target_column])
            print_info(f"Variable objetivo '{target_column}' separada")
        
        # Identificar tipos de variables
        self.numeric_features = df_work.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = df_work.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print_metric("Variables numéricas", len(self.numeric_features))
        print_metric("Variables categóricas", len(self.categorical_features))
        
        # Manejo de valores faltantes
        missing = df_work.isnull().sum()
        if missing.sum() > 0:
            print_warning(f"Valores faltantes detectados: {missing.sum()} total")
            
            # Imputar valores faltantes en numéricas (mediana)
            if len(self.numeric_features) > 0:
                imputer_num = SimpleImputer(strategy='median')
                df_work[self.numeric_features] = imputer_num.fit_transform(df_work[self.numeric_features])
            
            # Imputar valores faltantes en categóricas (moda)
            if len(self.categorical_features) > 0:
                imputer_cat = SimpleImputer(strategy='most_frequent')
                df_work[self.categorical_features] = imputer_cat.fit_transform(df_work[self.categorical_features])
            
            print_success("Valores faltantes imputados (mediana/moda)")
        else:
            print_success("No hay valores faltantes")
        
        # Codificación de variables categóricas
        if len(self.categorical_features) > 0:
            print_info(f"Codificando {len(self.categorical_features)} variables categóricas...")
            label_encoders = {}
            for col in self.categorical_features:
                le = LabelEncoder()
                df_work[col] = le.fit_transform(df_work[col].astype(str))
                label_encoders[col] = le
            print_success("Variables categóricas codificadas (Label Encoding)")
        
        # Normalización de datos
        print_info("Normalizando datos con StandardScaler...")
        self.df_processed = pd.DataFrame(
            self.scaler.fit_transform(df_work),
            columns=df_work.columns,
            index=df_work.index
        )
        print_success("Datos normalizados (μ=0, σ=1)")
        
        print_success(f"Preparación completada → Shape: {self.df_processed.shape}")
        return self
    
    # ============================================================================
    # 3. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
    # ============================================================================
    
    def exploratory_analysis(self, save_plots=True):
        """
        Realiza análisis exploratorio de datos con visualizaciones.
        
        Args:
            save_plots (bool): Guardar gráficos en archivos
        """
        print_header("📊 ANÁLISIS EXPLORATORIO (EDA)", Fore.MAGENTA)
        
        # Estadísticas descriptivas
        print_info("Estadísticas Descriptivas:")
        print(self.df.describe().to_string())
        
        # Información del dataset
        print_metric("Filas", self.df.shape[0])
        print_metric("Columnas", self.df.shape[1])
        print_metric("Memoria", f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f}", " MB")
        
        # Matriz de correlación
        if len(self.numeric_features) > 1:
            print_info("Generando matriz de correlación...")
            fig, ax = plt.subplots(figsize=(12, 8))
            corr_matrix = self.df[self.numeric_features].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, ax=ax)
            plt.title('Matriz de Correlación de Variables Numéricas', fontsize=14, fontweight='bold')
            plt.tight_layout()
            self._save_plot('correlation_matrix.png')
        
        # Distribuciones de variables numéricas
        if len(self.numeric_features) > 0:
            print_info("Generando distribuciones de variables...")
            n_features = len(self.numeric_features)
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = axes.flatten() if n_features > 1 else [axes]
            
            for idx, col in enumerate(self.numeric_features):
                axes[idx].hist(self.df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'Distribución: {col}', fontweight='bold')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frecuencia')
                axes[idx].grid(alpha=0.3)
            
            # Ocultar ejes vacíos
            for idx in range(n_features, len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            self._save_plot('distributions.png')
        
        # Detección de outliers (boxplots)
        if len(self.numeric_features) > 0:
            print_info("Detectando outliers con boxplots...")
            fig, axes = plt.subplots(1, 1, figsize=(14, 6))
            self.df[self.numeric_features].boxplot(ax=axes)
            axes.set_title('Detección de Outliers - Variables Numéricas', fontsize=14, fontweight='bold')
            axes.set_ylabel('Valores')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            self._save_plot('outliers_boxplot.png')
        
        print_success("Análisis exploratorio completado")
        return self
    
    # ============================================================================
    # 4. SIMPLIFICACIÓN DE DATOS (PCA)
    # ============================================================================
    
    def reduce_dimensions(self, n_components=None, variance_threshold=0.95):
        """
        Reduce dimensionalidad usando PCA.
        
        Args:
            n_components (int): Número de componentes (None para automático)
            variance_threshold (float): Varianza acumulada mínima a retener
        """
        print_header("🎯 SIMPLIFICACIÓN DE DATOS (PCA)", Fore.BLUE)
        
        if self.df_processed is None:
            print_error("Debe ejecutar prepare_data() primero")
            raise ValueError("Debe ejecutar prepare_data() primero")
        
        # Determinar número óptimo de componentes
        if n_components is None:
            pca_temp = PCA()
            pca_temp.fit(self.df_processed)
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= variance_threshold) + 1
            print_info(f"Componentes para {variance_threshold*100}% varianza: {n_components}")
        
        # Aplicar PCA
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        df_pca = self.pca.fit_transform(self.df_processed)
        
        # Crear DataFrame con componentes principales
        columns = [f'PC{i+1}' for i in range(n_components)]
        self.df_pca = pd.DataFrame(df_pca, columns=columns, index=self.df_processed.index)
        
        # Varianza explicada
        var_exp = self.pca.explained_variance_ratio_
        cum_var_exp = np.cumsum(var_exp)
        
        print_info("Varianza Explicada por Componente:")
        for i, (var, cum_var) in enumerate(zip(var_exp, cum_var_exp)):
            print(f"   PC{i+1}: {var*100:5.2f}% (Acumulada: {cum_var*100:5.2f}%)")
        
        # Visualización de varianza explicada
        print_info("Generando gráficos de varianza PCA...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Gráfico de barras
        ax1.bar(range(1, n_components+1), var_exp * 100, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Componente Principal', fontweight='bold')
        ax1.set_ylabel('Varianza Explicada (%)', fontweight='bold')
        ax1.set_title('Varianza Explicada por Componente', fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # Gráfico acumulado
        ax2.plot(range(1, n_components+1), cum_var_exp * 100, marker='o', linewidth=2)
        ax2.axhline(y=variance_threshold*100, color='r', linestyle='--', 
                   label=f'{variance_threshold*100}% Varianza')
        ax2.set_xlabel('Número de Componentes', fontweight='bold')
        ax2.set_ylabel('Varianza Explicada Acumulada (%)', fontweight='bold')
        ax2.set_title('Varianza Acumulada', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        self._save_plot('pca_variance.png')
        
        print_success(f"Reducción de {self.df_processed.shape[1]} a {n_components} dimensiones")
        return self
    
    # ============================================================================
    # 5. CLUSTERING K-MEANS
    # ============================================================================
    
    def find_optimal_k(self, k_range=range(2, 11), use_pca=True):
        """
        Encuentra el número óptimo de clusters usando método del codo y silueta.
        
        Args:
            k_range (range): Rango de valores K a probar
            use_pca (bool): Usar datos reducidos por PCA
        """
        print("\n🔎 BÚSQUEDA DEL K ÓPTIMO")
        print("=" * 60)
        
        data = self.df_pca if use_pca and hasattr(self, 'df_pca') else self.df_processed
        
        inertias = []
        silhouette_scores = []
        davies_bouldin_scores = []
        calinski_harabasz_scores = []
        
        print(f"🔄 Evaluando K de {min(k_range)} a {max(k_range)}...")
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(data)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(data, labels))
            davies_bouldin_scores.append(davies_bouldin_score(data, labels))
            calinski_harabasz_scores.append(calinski_harabasz_score(data, labels))
            
            print(f"   K={k}: Inercia={kmeans.inertia_:.2f}, "
                  f"Silueta={silhouette_scores[-1]:.3f}")
        
        # Método del codo automatizado
        from scipy.signal import savgol_filter
        try:
            smoothed = savgol_filter(inertias, window_length=min(5, len(inertias)), polyorder=2)
            diffs = np.diff(smoothed)
            optimal_idx = np.argmax(diffs[:-1] - diffs[1:]) + 1
            self.optimal_k = list(k_range)[optimal_idx]
        except:
            # Si falla, usar el mejor silhouette score
            self.optimal_k = list(k_range)[np.argmax(silhouette_scores)]
        
        print(f"\n🎯 K óptimo sugerido: {self.optimal_k}")
        
        # Visualización de métricas
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Método del codo
        axes[0, 0].plot(k_range, inertias, marker='o', linewidth=2)
        axes[0, 0].axvline(x=self.optimal_k, color='r', linestyle='--', 
                          label=f'K óptimo = {self.optimal_k}')
        axes[0, 0].set_xlabel('Número de Clusters (K)', fontweight='bold')
        axes[0, 0].set_ylabel('Inercia', fontweight='bold')
        axes[0, 0].set_title('Método del Codo', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Coeficiente de Silueta
        axes[0, 1].plot(k_range, silhouette_scores, marker='o', linewidth=2, color='green')
        axes[0, 1].axvline(x=self.optimal_k, color='r', linestyle='--', 
                          label=f'K óptimo = {self.optimal_k}')
        axes[0, 1].set_xlabel('Número de Clusters (K)', fontweight='bold')
        axes[0, 1].set_ylabel('Coeficiente de Silueta', fontweight='bold')
        axes[0, 1].set_title('Análisis de Silueta', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Davies-Bouldin Index (menor es mejor)
        axes[1, 0].plot(k_range, davies_bouldin_scores, marker='o', linewidth=2, color='orange')
        axes[1, 0].set_xlabel('Número de Clusters (K)', fontweight='bold')
        axes[1, 0].set_ylabel('Davies-Bouldin Index', fontweight='bold')
        axes[1, 0].set_title('Davies-Bouldin Index (menor es mejor)', fontweight='bold')
        axes[1, 0].grid(alpha=0.3)
        
        # Calinski-Harabasz Index (mayor es mejor)
        axes[1, 1].plot(k_range, calinski_harabasz_scores, marker='o', linewidth=2, color='purple')
        axes[1, 1].set_xlabel('Número de Clusters (K)', fontweight='bold')
        axes[1, 1].set_ylabel('Calinski-Harabasz Index', fontweight='bold')
        axes[1, 1].set_title('Calinski-Harabasz Index (mayor es mejor)', fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('optimal_k_analysis.png', dpi=300, bbox_inches='tight')
        print(f"💾 Guardado: optimal_k_analysis.png")
        plt.show()
        
        return self
    
    def fit_kmeans(self, n_clusters=None, use_pca=True):
        """
        Entrena el modelo K-means con el número de clusters especificado.
        
        Args:
            n_clusters (int): Número de clusters (None usa el óptimo encontrado)
            use_pca (bool): Usar datos reducidos por PCA
        """
        print("\n🤖 ENTRENAMIENTO K-MEANS")
        print("=" * 60)
        
        if n_clusters is None:
            n_clusters = self.optimal_k if self.optimal_k else 3
        
        data = self.df_pca if use_pca and hasattr(self, 'df_pca') else self.df_processed
        
        print(f"🎯 Entrenando K-means con K={n_clusters}...")
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        
        self.labels = self.kmeans.fit_predict(data)
        
        # Agregar etiquetas al DataFrame original
        self.df['Cluster'] = self.labels
        if use_pca and hasattr(self, 'df_pca'):
            self.df_pca['Cluster'] = self.labels
        self.df_processed['Cluster'] = self.labels
        
        # Métricas de evaluación
        silhouette = silhouette_score(data, self.labels)
        davies_bouldin = davies_bouldin_score(data, self.labels)
        calinski_harabasz = calinski_harabasz_score(data, self.labels)
        
        # Almacenar métricas para el informe
        self.metrics = {
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'calinski_harabasz': calinski_harabasz,
            'inertia': self.kmeans.inertia_,
            'n_clusters': n_clusters
        }
        
        print(f"\n📊 Métricas de Evaluación:")
        print(f"   🎯 Coeficiente de Silueta: {silhouette:.4f} (rango: -1 a 1, mejor cerca de 1)")
        print(f"   🎯 Davies-Bouldin Index: {davies_bouldin:.4f} (menor es mejor)")
        print(f"   🎯 Calinski-Harabasz Index: {calinski_harabasz:.4f} (mayor es mejor)")
        print(f"   🎯 Inercia: {self.kmeans.inertia_:.4f}")
        
        # Distribución de clusters
        print(f"\n📈 Distribución de Clusters:")
        cluster_counts = pd.Series(self.labels).value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            percentage = (count / len(self.labels)) * 100
            print(f"   Cluster {cluster_id}: {count} muestras ({percentage:.1f}%)")
        
        # Guardar resumen de métricas en archivo de texto
        self._save_metrics_summary()
        
        print(f"\n✅ Modelo K-means entrenado exitosamente")
        return self
    
    def _save_metrics_summary(self):
        """Guarda un resumen de las métricas de evaluación en archivo de texto."""
        if not self.metrics:
            return
        
        filepath = os.path.join(self.output_dir, 'metrics_summary.txt')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("RESUMEN DE MÉTRICAS - ANÁLISIS K-MEANS CLUSTERING\n")
            f.write("="*70 + "\n\n")
            f.write(f"Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("CONFIGURACIÓN DEL MODELO:\n")
            f.write(f"  - Número de clusters (K): {self.metrics['n_clusters']}\n")
            f.write(f"  - Semilla aleatoria: {self.random_state}\n\n")
            
            f.write("MÉTRICAS DE EVALUACIÓN:\n")
            f.write(f"  - Coeficiente de Silueta: {self.metrics['silhouette']:.6f}\n")
            f.write(f"    (Rango: -1 a 1, mejor cerca de 1)\n\n")
            
            f.write(f"  - Davies-Bouldin Index: {self.metrics['davies_bouldin']:.6f}\n")
            f.write(f"    (Menor es mejor, indica separación entre clusters)\n\n")
            
            f.write(f"  - Calinski-Harabasz Index: {self.metrics['calinski_harabasz']:.6f}\n")
            f.write(f"    (Mayor es mejor, ratio de dispersión)\n\n")
            
            f.write(f"  - Inercia: {self.metrics['inertia']:.6f}\n")
            f.write(f"    (Suma de distancias al cuadrado a los centroides)\n\n")
            
            f.write("INTERPRETACIÓN:\n")
            if self.metrics['silhouette'] > 0.5:
                f.write("  ✓ Excelente cohesión interna y separación entre clusters\n")
            elif self.metrics['silhouette'] > 0.3:
                f.write("  ✓ Buena estructura de clusters identificada\n")
            else:
                f.write("  ⚠ Los clusters podrían tener solapamiento\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print_success(f"Guardado: {self.output_dir}/metrics_summary.txt")

    
    # ============================================================================
    # 6. VISUALIZACIÓN DE RESULTADOS
    # ============================================================================
    
    def visualize_clusters(self, use_pca=True):
        """
        Visualiza los clusters en 2D y 3D.
        
        Args:
            use_pca (bool): Usar componentes principales para visualización
        """
        print("\n🎨 VISUALIZACIÓN DE CLUSTERS")
        print("=" * 60)
        
        if self.labels is None:
            raise ValueError("Debe ejecutar fit_kmeans() primero")
        
        # Preparar datos para visualización
        if use_pca and hasattr(self, 'df_pca'):
            data_viz = self.df_pca.drop('Cluster', axis=1)
            feature_names = [f'PC{i+1}' for i in range(data_viz.shape[1])]
        else:
            data_viz = self.df_processed.drop('Cluster', axis=1)
            feature_names = data_viz.columns.tolist()
        
        n_clusters = len(np.unique(self.labels))
        
        # Visualización 2D
        if data_viz.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            scatter = ax.scatter(
                data_viz.iloc[:, 0],
                data_viz.iloc[:, 1],
                c=self.labels,
                cmap='viridis',
                s=50,
                alpha=0.6,
                edgecolors='black',
                linewidth=0.5
            )
            
            # Plotear centroides
            if use_pca and hasattr(self, 'df_pca'):
                centroids = self.kmeans.cluster_centers_
            else:
                centroids = self.kmeans.cluster_centers_
            
            ax.scatter(
                centroids[:, 0],
                centroids[:, 1],
                c='red',
                marker='X',
                s=300,
                edgecolors='black',
                linewidth=2,
                label='Centroides'
            )
            
            ax.set_xlabel(feature_names[0], fontweight='bold', fontsize=12)
            ax.set_ylabel(feature_names[1], fontweight='bold', fontsize=12)
            ax.set_title(f'Visualización de {n_clusters} Clusters - 2D', 
                        fontweight='bold', fontsize=14)
            ax.legend()
            ax.grid(alpha=0.3)
            
            plt.colorbar(scatter, ax=ax, label='Cluster')
            plt.tight_layout()
            plt.savefig('clusters_2d.png', dpi=300, bbox_inches='tight')
            print("💾 Guardado: clusters_2d.png")
            plt.show()
        
        # Visualización 3D
        if data_viz.shape[1] >= 3:
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(
                data_viz.iloc[:, 0],
                data_viz.iloc[:, 1],
                data_viz.iloc[:, 2],
                c=self.labels,
                cmap='viridis',
                s=50,
                alpha=0.6,
                edgecolors='black',
                linewidth=0.5
            )
            
            # Plotear centroides
            ax.scatter(
                centroids[:, 0],
                centroids[:, 1],
                centroids[:, 2] if centroids.shape[1] >= 3 else 0,
                c='red',
                marker='X',
                s=300,
                edgecolors='black',
                linewidth=2,
                label='Centroides'
            )
            
            ax.set_xlabel(feature_names[0], fontweight='bold', fontsize=11)
            ax.set_ylabel(feature_names[1], fontweight='bold', fontsize=11)
            ax.set_zlabel(feature_names[2], fontweight='bold', fontsize=11)
            ax.set_title(f'Visualización de {n_clusters} Clusters - 3D', 
                        fontweight='bold', fontsize=14)
            ax.legend()
            
            plt.colorbar(scatter, ax=ax, label='Cluster', shrink=0.5)
            plt.tight_layout()
            plt.savefig('clusters_3d.png', dpi=300, bbox_inches='tight')
            print("💾 Guardado: clusters_3d.png")
            plt.show()
        
        print("\n✅ Visualización completada")
        return self
    
    def analyze_clusters(self):
        """
        Analiza las características de cada cluster.
        """
        print_header("📊 ANÁLISIS DE CLUSTERS", Fore.GREEN)
        
        if self.labels is None:
            print_error("Debe ejecutar fit_kmeans() primero")
            raise ValueError("Debe ejecutar fit_kmeans() primero")
        
        # Estadísticas por cluster (solo numéricas)
        print_info("Calculando estadísticas por cluster...")
        cluster_analysis = self.df.groupby('Cluster')[self.numeric_features].mean()
        
        print_info("Estadísticas Promedio por Cluster:")
        print(cluster_analysis.to_string())
        
        # Guardar análisis
        filepath = os.path.join(self.output_dir, 'cluster_analysis.csv')
        cluster_analysis.to_csv(filepath)
        print_success(f"Guardado: cluster_analysis.csv")
        
        # Visualización de características por cluster
        if len(self.numeric_features) > 0:
            print_info("Generando gráfico de características por cluster...")
            fig, ax = plt.subplots(figsize=(14, 6))
            cluster_analysis.plot(kind='bar', ax=ax)
            ax.set_title('Características Promedio por Cluster', fontweight='bold', fontsize=14)
            ax.set_xlabel('Cluster', fontweight='bold')
            ax.set_ylabel('Valor Promedio', fontweight='bold')
            ax.legend(title='Features', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=0)
            plt.tight_layout()
            self._save_plot('cluster_features.png')
        
        print_success("Análisis de clusters completado")
        return cluster_analysis
    
    def save_results(self, filename='clustered_data.csv'):
        """
        Guarda el dataset con las etiquetas de cluster asignadas.
        
        Args:
            filename (str): Nombre del archivo de salida
        """
        filepath = os.path.join(self.output_dir, filename)
        self.df.to_csv(filepath, index=False)
        print_success(f"Datos con clusters guardados: {filename}")
        return self
    
    def generate_report(self):
        """
        Genera un reporte HTML con todos los resultados.
        """
        print_header("� GENERANDO REPORTE HTML", Fore.CYAN)
        
        report_file = os.path.join(self.output_dir, 'reporte_kmeans.html')
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Reporte K-means Clustering</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    margin-top: 30px;
                }}
                .metric {{
                    background-color: white;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .plot {{
                    margin: 20px 0;
                    text-align: center;
                }}
                img {{
                    max-width: 100%;
                    border-radius: 5px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .timestamp {{
                    color: #7f8c8d;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <h1>🎯 Reporte de Análisis K-means Clustering</h1>
            <p class="timestamp">Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="metric">
                <h2>📊 Información del Dataset</h2>
                <p><strong>Filas:</strong> {self.df.shape[0]}</p>
                <p><strong>Columnas:</strong> {self.df.shape[1]}</p>
                <p><strong>Clusters encontrados:</strong> {len(np.unique(self.labels))}</p>
            </div>
            
            <h2>📈 Visualizaciones</h2>
        """
        
        # Agregar todos los gráficos generados
        for plot_path in self.plots_generated:
            plot_name = os.path.basename(plot_path)
            html_content += f"""
            <div class="plot">
                <h3>{plot_name.replace('_', ' ').replace('.png', '').title()}</h3>
                <img src="{plot_name}" alt="{plot_name}">
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print_success(f"Reporte HTML generado: {report_file}")
        print_info(f"Abre el archivo '{report_file}' en tu navegador")
        
        return report_file
    
    def generate_markdown_report(self):
        """
        Genera el informe oficial en formato Markdown según requerimientos AA2-EV01.
        """
        print_header("📝 GENERANDO INFORME MARKDOWN (AA2-EV01)", Fore.GREEN)
        
        report_file = 'AA2-EV01_Informe_KMeans.md'
        
        # Preparar contenido del informe
        markdown_content = f"""# Informe de Análisis K-means Clustering
## Actividad AA2-EV01: Algoritmo de Agrupamiento No Supervisado con Python

---

### 📋 Información General

- **Autor:** Sistema de Análisis ML
- **Fecha de ejecución:** {datetime.now().strftime('%d de %B de %Y, %H:%M:%S')}
- **Programa:** SENA - Algoritmo de Agrupamiento No Supervisado con Python
- **Actividad:** AA2-EV01

---

### 🎯 Resultados del Análisis

#### Configuración del Modelo

- **Número óptimo de clusters encontrado:** {self.metrics.get('n_clusters', 'N/A')}
- **Semilla aleatoria utilizada:** {self.random_state}
- **Método de reducción dimensional:** {"PCA aplicado" if hasattr(self, 'df_pca') else "Sin PCA"}
- **Tamaño del dataset:** {self.df.shape[0]} muestras × {self.df.shape[1]} variables

---

### 📊 Métricas de Evaluación del Modelo

Las siguientes métricas evalúan la calidad del agrupamiento obtenido:

#### 1. Coeficiente de Silueta
- **Valor obtenido:** {self.metrics.get('silhouette', 0):.6f}
- **Rango:** -1 a 1 (valores cercanos a 1 son mejores)
- **Interpretación:** {"Excelente cohesión y separación" if self.metrics.get('silhouette', 0) > 0.5 else "Buena estructura identificada" if self.metrics.get('silhouette', 0) > 0.3 else "Estructura moderada"}

#### 2. Índice Davies-Bouldin
- **Valor obtenido:** {self.metrics.get('davies_bouldin', 0):.6f}
- **Interpretación:** Menor es mejor (mide separación entre clusters)
- **Resultado:** {"Excelente separación" if self.metrics.get('davies_bouldin', 1) < 0.8 else "Buena separación" if self.metrics.get('davies_bouldin', 1) < 1.2 else "Separación aceptable"}

#### 3. Índice Calinski-Harabasz
- **Valor obtenido:** {self.metrics.get('calinski_harabasz', 0):.6f}
- **Interpretación:** Mayor es mejor (ratio de dispersión entre/dentro clusters)
- **Resultado:** {"Muy bien definidos" if self.metrics.get('calinski_harabasz', 0) > 200 else "Bien definidos" if self.metrics.get('calinski_harabasz', 0) > 100 else "Moderadamente definidos"}

#### 4. Inercia (Within-Cluster Sum of Squares)
- **Valor obtenido:** {self.metrics.get('inertia', 0):.6f}
- **Interpretación:** Suma de distancias al cuadrado de cada muestra a su centroide

---

### 📁 Archivos Generados

Todos los resultados se encuentran en la carpeta `{self.output_dir}/`:

#### Visualizaciones (PNG)
"""
        
        # Listar archivos PNG generados
        png_files = [os.path.basename(f) for f in self.plots_generated if f.endswith('.png')]
        for idx, png_file in enumerate(png_files, 1):
            markdown_content += f"{idx}. `{png_file}`\n"
        
        # Listar archivos CSV y TXT
        markdown_content += f"""
#### Archivos de Datos y Métricas
"""
        data_files = []
        for file in os.listdir(self.output_dir):
            if file.endswith(('.csv', '.txt')):
                data_files.append(file)
        
        for idx, data_file in enumerate(data_files, 1):
            markdown_content += f"{idx}. `{data_file}`\n"
        
        # Agregar resumen y conclusiones
        markdown_content += f"""
---

### 📝 Resumen del Análisis

El análisis de agrupamiento no supervisado con K-means identificó **{self.metrics.get('n_clusters', 'N/A')} clusters óptimos** en el dataset.

#### Calidad del Agrupamiento

"""
        
        # Generar resumen automático basado en métricas
        silhouette = self.metrics.get('silhouette', 0)
        davies_bouldin = self.metrics.get('davies_bouldin', 1)
        
        if silhouette > 0.5 and davies_bouldin < 0.8:
            summary = """El análisis mostró una **excelente cohesión interna** y **separación clara entre grupos**. Los clusters identificados presentan características distintivas y bien definidas, lo que indica que el algoritmo K-means logró identificar patrones significativos en los datos.

La configuración óptima del modelo permite segmentar efectivamente las observaciones en grupos homogéneos internamente pero heterogéneos entre sí."""
        elif silhouette > 0.3:
            summary = """El análisis demostró una **buena estructura de agrupamiento** con cohesión aceptable dentro de cada cluster y separación razonable entre grupos. Los clusters identificados muestran patrones distinguibles que pueden ser útiles para la comprensión del dataset.

Los resultados sugieren que existe una estructura natural en los datos que el algoritmo K-means pudo capturar adecuadamente."""
        else:
            summary = """El análisis identificó una **estructura moderada** en los datos. Si bien los clusters presentan cierto grado de separación, existe algún solapamiento entre grupos que podría requerir análisis adicional o ajuste de parámetros.

Se recomienda explorar diferentes valores de K o considerar métodos de clustering alternativos para mejorar la separación."""
        
        markdown_content += summary + "\n\n"
        
        markdown_content += f"""
#### Distribución de Muestras

"""
        # Agregar distribución de clusters
        if self.labels is not None:
            cluster_counts = pd.Series(self.labels).value_counts().sort_index()
            for cluster_id, count in cluster_counts.items():
                percentage = (count / len(self.labels)) * 100
                markdown_content += f"- **Cluster {cluster_id}:** {count} muestras ({percentage:.1f}%)\n"
        
        markdown_content += f"""
---

### 🔍 Metodología Aplicada

El pipeline de análisis siguió las siguientes etapas del aprendizaje automático:

1. **Selección y Carga de Datos:** Preparación del dataset de entrada
2. **Preparación de Datos:** 
   - Manejo de valores faltantes mediante imputación
   - Codificación de variables categóricas
   - Normalización con StandardScaler (μ=0, σ=1)
3. **Análisis Exploratorio (EDA):**
   - Estadísticas descriptivas
   - Matriz de correlación
   - Detección de outliers
4. **Simplificación de Datos:** {"Reducción dimensional con PCA" if hasattr(self, 'df_pca') else "Datos originales sin reducción"}
5. **Búsqueda de K Óptimo:** 
   - Método del codo
   - Análisis de silueta
   - Métricas comparativas
6. **Entrenamiento del Modelo:** K-means con parámetros optimizados
7. **Evaluación y Visualización:** Generación de gráficos y métricas

---

### 📚 Conclusiones

El análisis de agrupamiento no supervisado mediante el algoritmo K-means ha sido completado exitosamente, cumpliendo con los objetivos de la actividad AA2-EV01:

✅ **Pipeline completo implementado** con todas las etapas del aprendizaje automático

✅ **Métricas de evaluación calculadas** para validar la calidad del agrupamiento

✅ **Visualizaciones generadas** para facilitar la interpretación de resultados

✅ **Documentación completa** de metodología y resultados

Los clusters identificados pueden utilizarse para:
- Segmentación de datos
- Identificación de patrones
- Análisis exploratorio avanzado
- Toma de decisiones basada en grupos homogéneos

---

### 📖 Referencias

- Scikit-learn Documentation: K-means Clustering
- MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations"
- Rousseeuw, P. J. (1987). "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis"

---

**Fin del Informe AA2-EV01**

*Generado automáticamente por el pipeline de K-means Clustering*
"""
        
        # Guardar el informe en la raíz del proyecto
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print_success(f"Informe generado: {report_file}")
        print_info("El informe Markdown ha sido guardado en la raíz del proyecto")
        
        return report_file


# ============================================================================
# FUNCIÓN PRINCIPAL - PIPELINE COMPLETO
# ============================================================================

def run_kmeans_pipeline(filepath=None, data=None, n_clusters=None, use_pca=True, generate_html=True):
    """
    Ejecuta el pipeline completo de análisis K-means.
    
    Args:
        filepath (str): Ruta al archivo CSV
        data (DataFrame): DataFrame de pandas
        n_clusters (int): Número de clusters (None para búsqueda automática)
        use_pca (bool): Aplicar reducción de dimensionalidad
        generate_html (bool): Generar reporte HTML interactivo
    
    Returns:
        KMeansAnalyzer: Objeto con el análisis completo
    """
    print_header("🚀 PIPELINE DE ANÁLISIS K-MEANS", Fore.GREEN)
    
    analyzer = KMeansAnalyzer(random_state=42)
    
    # 1. Cargar datos
    analyzer.load_data(filepath=filepath, data=data)
    
    # 2. Preparar datos
    analyzer.prepare_data()
    
    # 3. Análisis exploratorio
    analyzer.exploratory_analysis()
    
    # 4. Reducción de dimensionalidad (opcional)
    if use_pca:
        analyzer.reduce_dimensions(variance_threshold=0.95)
    
    # 5. Encontrar K óptimo
    analyzer.find_optimal_k()
    
    # 6. Entrenar K-means
    analyzer.fit_kmeans(n_clusters=n_clusters, use_pca=use_pca)
    
    # 7. Visualizar resultados
    analyzer.visualize_clusters(use_pca=use_pca)
    
    # 8. Analizar clusters
    analyzer.analyze_clusters()
    
    # 9. Guardar resultados
    analyzer.save_results()
    
    # 10. Generar reporte HTML
    if generate_html:
        analyzer.generate_report()
    
    # 11. Generar informe Markdown (AA2-EV01)
    analyzer.generate_markdown_report()
    
    print_header("✅ PIPELINE COMPLETADO EXITOSAMENTE", Fore.GREEN)
    print_success(f"Resultados guardados en: {analyzer.output_dir}/")
    print_success(f"Total de gráficos generados: {len(analyzer.plots_generated)}")
    
    return analyzer


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    """
    Ejemplo de uso del script con datos sintéticos.
    Para usar tus propios datos, reemplaza con: run_kmeans_pipeline(filepath='tu_archivo.csv')
    """
    
    print("\n🎯 Generando datos de ejemplo...")
    
    # Crear dataset sintético para demostración
    from sklearn.datasets import make_blobs
    
    X, y_true = make_blobs(
        n_samples=300,
        centers=4,
        n_features=5,
        cluster_std=1.0,
        random_state=42
    )
    
    df_example = pd.DataFrame(
        X,
        columns=[f'Feature_{i+1}' for i in range(X.shape[1])]
    )
    
    # Agregar algunas variables categóricas
    df_example['Category_A'] = np.random.choice(['Tipo1', 'Tipo2', 'Tipo3'], size=len(df_example))
    df_example['Category_B'] = np.random.choice(['Grupo_X', 'Grupo_Y'], size=len(df_example))
    
    # Agregar algunos valores faltantes para demostrar imputación
    mask = np.random.random(df_example.shape) < 0.05
    df_example = df_example.mask(mask)
    
    print(f"✅ Dataset de ejemplo creado: {df_example.shape[0]} filas, {df_example.shape[1]} columnas")
    
    # Ejecutar pipeline completo
    analyzer = run_kmeans_pipeline(data=df_example, use_pca=True)
    
    print("\n" + "="*60)
    print("📚 INSTRUCCIONES DE USO:")
    print("="*60)
    print("""
    Para usar con tus propios datos CSV:
    
    1. Opción Simple:
       analyzer = run_kmeans_pipeline(filepath='tus_datos.csv')
    
    2. Opción Avanzada (paso a paso):
       analyzer = KMeansAnalyzer()
       analyzer.load_data(filepath='tus_datos.csv')
       analyzer.prepare_data(drop_columns=['columna_id'])
       analyzer.exploratory_analysis()
       analyzer.reduce_dimensions()
       analyzer.find_optimal_k()
       analyzer.fit_kmeans(n_clusters=3)
       analyzer.visualize_clusters()
       analyzer.analyze_clusters()
       analyzer.save_results()
    
    3. Personalizar parámetros:
       analyzer.find_optimal_k(k_range=range(2, 8))
       analyzer.reduce_dimensions(n_components=3)
       analyzer.fit_kmeans(n_clusters=5, use_pca=False)
    """)
