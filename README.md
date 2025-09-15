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
* * **`Iris_ML`**: Archivo que contiene el script del proyecto en formato .py.
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

# Flujo de Trabajo del Modelo de Regresión Lineal

Los siguientes items hacen referencia a los pasos que se realizaron para la aplicación del modelo de regresión lineal antes de transformar las predicciones en clasificaciones.

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
   Graficar un diagrama de dispersión y un gráfico de residuos para representar la relación entre valores reales y predichos, así como para evaluar la precisión del modelo.

8. **Evaluar las métricas de predicción**  
   Calcular las métricas de rendimiento del modelo, incluyendo:  
   - Error Cuadrático Medio (MSE)  
   - Error Absoluto Medio (MAE)  
   - Coeficiente de Determinación (R²)  

-----------------------------------------------------------------

## Clasificación de Especies

Debido a que el objetivo del proyecto es poder determinar qué planta será según los features, se debe redondear sus resultados que son números continuos, a valores de 0("setosa"), 1("versicolor"), 2("virginica"). Debe devolver la precisión de la predicción realizada.  

En este caso se usó la función .rint() para redondear los datos del conjunto de datos al entero más cercano, con la particularidad de que estos se mantienen como un número de tipo float().  

En el caso de .clip(), se usó esta función para limitar los valores de este conjunto de datos (en un array) dentro de un rango específico, se estableció un valor mínimo de 0 y valor máximo de 2, al realizar esta delimitación se asegura que las predicciones no se salgan del rango de clases establecido (0=setosa , 1=versicolor y 2=virginica).  

## K-fold-cross-validation

Esta técnica se usa para poder estudiar la precisión real del modelo, pues la respuesta de una regresión lineal es un número continuo, y debe ser una clase para determinar cual planta es.

Al realizarse un entrenamiento normal, la precisión puede llegar a ser exacta, pero no demuestra la capacidad real del modelo, sino la capacidad bajo los valores determinados por el entrenamiento. En este caso, esta técnica realiza múltiples repartos y pruebas para finalmente promediar los resultados. Esta técnica funciona de la siguiente manera: 

1. Se divide el dataset en K grupos o partes (pliegues) iguales.
2. Se repite K veces: entrena el modelo con K-1 pliegues y lo prueba con el pliegue restante.
3. Promedia las K puntuaciones de rendimiento obtenidas para obtener una evaluación final más fiable.

De esta manera se reduce el riesgo de que la variabilidad del muestreo en una única división de datos, dé una idea equivocada del verdadero rendimiento del modelo.

## Análisis de Resultados













