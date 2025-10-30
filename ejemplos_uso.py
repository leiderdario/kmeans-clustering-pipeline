"""
Ejemplo de Uso - K-means Clustering Pipeline
============================================
Este script demuestra diferentes formas de usar el pipeline de K-means.
"""

from kmeans_clustering import run_kmeans_pipeline, KMeansAnalyzer
import pandas as pd

# =============================================================================
# EJEMPLO 1: Uso Básico con Datos de Ejemplo
# =============================================================================

def ejemplo_basico():
    """Ejecuta el pipeline completo con datos CSV."""
    print("\n" + "="*60)
    print("EJEMPLO 1: Pipeline Completo con CSV")
    print("="*60)
    
    analyzer = run_kmeans_pipeline(
        filepath='ejemplo_datos.csv',
        use_pca=True,
        generate_html=True
    )
    
    print(f"\n✅ K óptimo encontrado: {analyzer.optimal_k}")
    print(f"✅ Archivos generados: {len(analyzer.plots_generated)}")
    return analyzer


# =============================================================================
# EJEMPLO 2: Control Paso a Paso
# =============================================================================

def ejemplo_personalizado():
    """Control manual de cada paso del análisis."""
    print("\n" + "="*60)
    print("EJEMPLO 2: Control Paso a Paso")
    print("="*60)
    
    # Crear analizador
    analyzer = KMeansAnalyzer(
        random_state=42,
        output_dir='mi_analisis_custom'
    )
    
    # Cargar datos
    analyzer.load_data(filepath='ejemplo_datos.csv')
    
    # Preparar datos (eliminar columnas innecesarias)
    analyzer.prepare_data(drop_columns=['Region'])
    
    # Análisis exploratorio
    analyzer.exploratory_analysis()
    
    # PCA con parámetros personalizados
    analyzer.reduce_dimensions(
        n_components=3,  # Forzar 3 componentes
        variance_threshold=0.90
    )
    
    # Buscar K óptimo en rango específico
    analyzer.find_optimal_k(k_range=range(2, 6))
    
    # Entrenar con K específico
    analyzer.fit_kmeans(n_clusters=3, use_pca=True)
    
    # Visualizar
    analyzer.visualize_clusters(use_pca=True)
    
    # Analizar y guardar
    cluster_stats = analyzer.analyze_clusters()
    analyzer.save_results('resultados_custom.csv')
    analyzer.generate_report()
    
    print(f"\n✅ Análisis completado")
    return analyzer, cluster_stats


# =============================================================================
# EJEMPLO 3: Usando DataFrame en Memoria
# =============================================================================

def ejemplo_dataframe():
    """Usar un DataFrame de pandas directamente."""
    print("\n" + "="*60)
    print("EJEMPLO 3: Con DataFrame de Pandas")
    print("="*60)
    
    # Crear o cargar datos
    df = pd.read_csv('ejemplo_datos.csv')
    
    # Filtrar o transformar datos según necesites
    df_filtrado = df[df['Age'] >= 25].copy()
    
    analyzer = run_kmeans_pipeline(
        data=df_filtrado,
        n_clusters=4,  # K fijo
        use_pca=False,  # Sin PCA
        generate_html=True
    )
    
    return analyzer


# =============================================================================
# EJEMPLO 4: Análisis Comparativo (Múltiples K)
# =============================================================================

def ejemplo_comparativo():
    """Comparar resultados con diferentes valores de K."""
    print("\n" + "="*60)
    print("EJEMPLO 4: Análisis Comparativo")
    print("="*60)
    
    from sklearn.metrics import silhouette_score
    
    df = pd.read_csv('ejemplo_datos.csv')
    resultados = {}
    
    for k in range(2, 7):
        print(f"\n🔄 Probando K={k}...")
        
        analyzer = KMeansAnalyzer(
            output_dir=f'resultados_k{k}'
        )
        
        analyzer.load_data(data=df)
        analyzer.prepare_data()
        analyzer.reduce_dimensions()
        analyzer.fit_kmeans(n_clusters=k, use_pca=True)
        
        # Guardar métricas
        resultados[k] = {
            'silhouette': silhouette_score(
                analyzer.df_pca.drop('Cluster', axis=1),
                analyzer.labels
            ),
            'inertia': analyzer.kmeans.inertia_,
            'analyzer': analyzer
        }
    
    # Mostrar comparación
    print("\n📊 COMPARACIÓN DE RESULTADOS:")
    print("-" * 60)
    for k, metrics in resultados.items():
        print(f"K={k}: Silueta={metrics['silhouette']:.4f}, "
              f"Inercia={metrics['inertia']:.2f}")
    
    return resultados


# =============================================================================
# EJEMPLO 5: Integración con Workflow Existente
# =============================================================================

def ejemplo_workflow():
    """Integrar K-means en un flujo de trabajo más grande."""
    print("\n" + "="*60)
    print("EJEMPLO 5: Integración en Workflow")
    print("="*60)
    
    # 1. Cargar y preprocesar datos
    df = pd.read_csv('ejemplo_datos.csv')
    
    # 2. Tu lógica de negocio aquí
    # ... procesamiento adicional ...
    
    # 3. Ejecutar clustering
    analyzer = KMeansAnalyzer()
    analyzer.load_data(data=df)
    analyzer.prepare_data()
    analyzer.reduce_dimensions()
    analyzer.find_optimal_k()
    analyzer.fit_kmeans(use_pca=True)
    
    # 4. Obtener resultados para uso posterior
    df_con_clusters = analyzer.df.copy()
    centroides = analyzer.kmeans.cluster_centers_
    
    # 5. Usar resultados en tu aplicación
    print(f"\n✅ Datos con {analyzer.optimal_k} clusters asignados")
    print(f"📊 Distribución:")
    print(df_con_clusters['Cluster'].value_counts())
    
    return df_con_clusters, centroides


# =============================================================================
# EJECUCIÓN DE EJEMPLOS
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("\n🎯 EJEMPLOS DE USO - K-MEANS CLUSTERING PIPELINE")
    print("="*60)
    print("\nSelecciona un ejemplo:")
    print("1. Pipeline completo básico")
    print("2. Control paso a paso personalizado")
    print("3. Usar DataFrame de pandas")
    print("4. Análisis comparativo (múltiples K)")
    print("5. Integración en workflow")
    print("0. Ejecutar todos los ejemplos")
    
    try:
        opcion = input("\nIngresa el número de ejemplo (0-5): ").strip()
        
        if opcion == "1":
            ejemplo_basico()
        elif opcion == "2":
            ejemplo_personalizado()
        elif opcion == "3":
            ejemplo_dataframe()
        elif opcion == "4":
            ejemplo_comparativo()
        elif opcion == "5":
            ejemplo_workflow()
        elif opcion == "0":
            print("\n🚀 Ejecutando todos los ejemplos...")
            ejemplo_basico()
            input("\nPresiona Enter para continuar...")
            ejemplo_personalizado()
            input("\nPresiona Enter para continuar...")
            ejemplo_dataframe()
            input("\nPresiona Enter para continuar...")
            ejemplo_comparativo()
            input("\nPresiona Enter para continuar...")
            ejemplo_workflow()
        else:
            print("❌ Opción inválida. Ejecutando ejemplo básico...")
            ejemplo_basico()
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Ejecución cancelada por el usuario.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ EJEMPLOS COMPLETADOS")
    print("="*60)
    print("\n💡 Tip: Revisa los directorios 'resultados_kmeans/' y ")
    print("   'mi_analisis_custom/' para ver los archivos generados.")
