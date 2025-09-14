# Modelo de regresión Lineal aplicado al dataset de Iris de Sci-kit Learn
El siguiente proyecto presenta el desarrollo de un modelo de Regresión lineal para el dataset Iris de scikit-learn (sklearn). Iris es un dataset de juguete empleado en el Machine Learning gracias a su simplicidad, ya que es pequeño, limpio y fácil de entender. El dataset de Iris contiene un conjunto de datos con 150 muestras de flores Iris, las cuales están divididas de forma equitativa en tres especies diferentes, hay un total de 50 muestras por cada una de las especies. 

## Características del Proyecto

La variable objetivo en este caso son las siguientes especies de flor Iris: 

<div align="center">
  
| Especies         | Imagen de Referencia | 
|-----------------|-----------------|
| **Iris Setosa** | <img src="https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg" width="25%"> |
| **Iris Virginica** | <img src="https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg" width="25%"> |
| **Iris Versicolor** | <img width="25%" alt="image" src="https://github.com/user-attachments/assets/1b0dce5a-0807-402a-befb-87e5a2416d59" />|

</div>

Para cada una de estas especies, se comprenden las siguientes features: 

- Largo del sépalo (cm)
- Ancho del sépalo (cm)
- Largo del pétalo (cm)
- Ancho del pétalo (cm)

**Diferencias en las Medidas**  
La mejor manera de ver las diferencias es comparando las distribuciones de cada medida (largo y ancho de sépalo y pétalo) para cada una de las tres especies.

1. Iris Setosa: Generalmente tiene los pétalos más pequeños y los sépalos más anchos pero más cortos en comparación con las otras dos.  

2. Iris Versicolor: Sus medidas suelen estar en un rango intermedio entre Setosa y Virginica.

3. Iris Virginica: Tiende a tener los pétalos y sépalos más grandes (tanto en largo como en ancho).

## Librerías utilizadas

| Librería / Módulo | Descripción |
|-------------------|-------------|
| `sklearn.datasets` (`load_iris`) | Contiene datasets de ejemplo, como **Iris**, muy usado para pruebas de clasificación y regresión. |
| `pandas` | Librería para la **manipulación y análisis de datos** mediante estructuras como DataFrames y Series. |
| `sklearn.model_selection` (`train_test_split`) | Herramienta para **dividir datasets** en subconjuntos de entrenamiento y prueba. |
| `sklearn.preprocessing` (`StandardScaler`, `MinMaxScaler`) | Funciones para **escalar/normalizar datos** y mejorar el rendimiento de los modelos. |
| `sklearn.linear_model` (`LinearRegression`) | Implementación de modelos de **regresión lineal**. |
| `sklearn.metrics` (`mean_absolute_error`, `mean_squared_error`, `r2_score`, `f1_score`) | Métricas para **evaluar el rendimiento** de los modelos. |
| `seaborn` | Librería de **visualización estadística** basada en matplotlib, ideal para gráficos avanzados. |
| `matplotlib.pyplot` | Librería para **crear gráficos y visualizaciones** en 2D de forma flexible. |

## Estructura del proyecto

* **`Linear_Model_Iris_DSMatallana_LELatorre.ipynb`**: Notebook de Google Colab con todo el código fuente, desde la carga de datos hasta la evaluación final.
* **`requirements.txt`**: Archivo que lista todas las dependencias de Python para una fácil instalación del entorno.
* **`README.md`**: Documentación del proyecto.

---

## Instalación y uso

Para replicar este proyecto en su entorno local, siga estos pasos:  

1.  **Clone el repositorio:**
    ```bash
    git clone https://github.com/Lenna888/Linear_Model_Iris.git
    cd Linear_Model_Iris
    ```

2.  **Cree un entorno virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instale las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **O ejecute el Notebook:**
    Abre el archivo `Linear_Model_Iris_DSMatallana_LELatorre.ipynb` en Jupyter Notebook, JupyterLab o Google Colab para ver y ejecutar el análisis.

---

# Flujo de Trabajo del Modelo de regresión Lineal
-----------------------------------------------------------------

1. **Cargar el dataset de Iris**  
   Importar el dataset de Iris en el entorno para que esté disponible para el preprocesamiento 
   y el entrenamiento del modelo.

2. **Crear las características (features) y la variable objetivo**  
   Definir las variables independientes (features) y la variable dependiente (objetivo) 
   que se usarán para entrenar el modelo de regresión.

3. **Dividir el conjunto de datos**  
   Separar el dataset en dos subconjuntos: entrenamiento (80%) y prueba (20%).  
   Se establece una semilla aleatoria para garantizar la reproducibilidad.

4. **Normalizar el conjunto de datos**  
   Aplicar escalado de características a los conjuntos de entrenamiento y prueba usando uno de los siguientes escaladores:  
   - **StandardScaler**: estandariza las características con media 0 y desviación estándar 1.  
   - **MinMaxScaler**: transforma las características escalándolas en un rango entre 0 y 1.  

5. **Entrenar el modelo con los datos normalizados**  
   Ajustar un modelo de regresión lineal utilizando el conjunto de entrenamiento normalizado elegido.

6. **Realizar predicciones**  
   Usar el modelo de regresión entrenado para generar predicciones sobre el conjunto de prueba normalizado.

7. **Visualizar los resultados**  
   Graficar un diagrama de dispersión y un gráfico de residuos para representar la relación entre valores reales y predichos, 
   así como para evaluar la precisión del modelo.

8. **Evaluar las métricas de predicción**  
   Calcular las métricas de rendimiento del modelo, incluyendo:  
   - Error Cuadrático Medio (MSE)  
   - Error Absoluto Medio (MAE)  
   - Coeficiente de Determinación (R²)  

-----------------------------------------------------------------









