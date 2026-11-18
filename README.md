# Detección de Peligros Industriales mediante Visión por Computador

**Proyecto académico | Inteligencia Artificial II**

Este repositorio documenta el desarrollo y la evaluación comparativa de modelos de detección de objetos orientados a la identificación de peligros en entornos industriales. Se comparan cuatro variantes de la familia YOLO bajo un protocolo experimental común.

## Material del proyecto

| Recurso | Enlace |
|---|---|
| **Presentación académica** | [Consultar diapositivas](https://docs.google.com/presentation/d/1e-HPYCyndG7kAM-_mRlLrNnnxmdzKGZv/edit?usp=sharing&ouid=110979037433739738574&rtpof=true&sd=true) |
| **Notebook ejecutable en Kaggle** | [Comparación justa YOLO11 y YOLO26](https://www.kaggle.com/code/andreafernandabernal/iaii-final-comparacion-justa-yolo11-yolo26) |
| **Notebook incluido en este repositorio** | [`notebooks/iaii-final-comparacion-justa-yolo11-yolo26.ipynb`](notebooks/iaii-final-comparacion-justa-yolo11-yolo26.ipynb) |

## 1. Resumen

El proyecto aborda la detección automática de cinco categorías de peligros industriales a partir de imágenes. El trabajo realiza fine-tuning y compara los siguientes detectores:

- YOLO11n
- YOLO11s
- YOLO26n
- YOLO26s

La comparación considera métricas de detección, rendimiento por clase, tiempo de inferencia, número de parámetros y tamaño del checkpoint. También incluye un análisis adicional del mejor modelo YOLO26 con inferencia con NMS y end-to-end sin NMS.

## 2. Objetivos

### Objetivo general
Evaluar comparativamente variantes de YOLO para la detección de peligros industriales, utilizando un protocolo experimental común que permita analizar sus resultados sobre un conjunto de prueba independiente.

### Objetivos específicos
- Preparar las particiones de entrenamiento, validación y prueba.
- Ajustar cuatro variantes de YOLO bajo los mismos hiperparámetros principales.
- Medir el desempeño global y por clase en el conjunto de prueba.
- Comparar precisión, recall, mAP y velocidad de inferencia.
- Examinar el compromiso entre desempeño, complejidad del modelo y costo de inferencia.
- Analizar el efecto de NMS frente al modo end-to-end en el mejor YOLO26 evaluado.

## 3. Dataset

El notebook utiliza el conjunto **Industrial Hazards Detection**, con 1.343 imágenes y cinco clases:

| Clase | Nombre original |
|---|---|
| 1 | `chemical hazard` |
| 2 | `fire` |
| 3 | `no helmet` |
| 4 | `smoke` |
| 5 | `water leak` |

### Protocolo de particionamiento

Para evitar utilizar el mismo conjunto en la selección del checkpoint y en el reporte final, el `train` original se divide de manera determinista con semilla `42`:

- **80 %** del `train` original: entrenamiento.
- **20 %** del `train` original: validación y early stopping.
- **`valid` original**: conjunto de prueba (*test*), reservado para la evaluación final.

Solo se consideran imágenes que cuentan con su archivo de etiquetas correspondiente. El notebook verifica la presencia de las cinco clases en las tres particiones. Los conteos de distribución corresponden a **objetos anotados**, no a imágenes.

> El dataset no se redistribuye en este repositorio. Debe obtenerse desde su fuente original y agregarse como entrada en Kaggle, respetando sus condiciones de uso.

## 4. Metodología experimental

Los cuatro modelos se entrenan con la misma configuración principal:

| Parámetro | Valor |
|---|---:|
| Pesos iniciales | Preentrenados |
| Tamaño de entrada | 640 × 640 |
| Optimizador | AdamW |
| Tasa de aprendizaje inicial (`lr0`) | 0.001 |
| `weight_decay` | 0.0005 |
| Épocas máximas | 80 |
| Batch size | 16 |
| Early stopping (`patience`) | 15 |
| Semilla | 42 |

El procedimiento para cada modelo es:

1. Cargar los pesos preentrenados.
2. Entrenar utilizando únicamente `train` y `val`.
3. Seleccionar el checkpoint `best.pt` según la validación.
4. Evaluar el checkpoint seleccionado sobre `test`.
5. Registrar métricas globales, métricas por clase y artefactos en Weights & Biases.

La diferencia principal entre experimentos corresponde a la arquitectura y escala del detector.

## 5. Resultados principales

Los siguientes valores corresponden a la salida registrada en el notebook adjunto. La métrica utilizada para ordenar los modelos es `mAP@50:95` sobre el conjunto `test`.

| Modelo | Precision | Recall | mAP@50 | mAP@50:95 | Inferencia (ms/imagen) |
|---|---:|---:|---:|---:|---:|
| YOLO11n | 0.6410 | 0.6599 | 0.6603 | **0.3339** | **4.7078** |
| YOLO26s | **0.7478** | 0.5952 | 0.6415 | 0.3319 | 12.7907 |
| YOLO26n | 0.7430 | 0.6303 | 0.6586 | 0.3308 | 4.8972 |
| YOLO11s | 0.6814 | 0.5859 | 0.6051 | 0.3082 | 12.2700 |

### Lectura de los resultados

- **Mayor mAP@50:95:** YOLO11n, con 0.3339.
- **Mayor precisión:** YOLO26s, con 0.7478.
- **Mayor recall:** YOLO11n, con 0.6599.
- **Menor tiempo de inferencia reportado:** YOLO11n, con 4.7078 ms/imagen.

La selección de un modelo para un escenario industrial no debe basarse exclusivamente en una métrica. Es necesario considerar el costo de los falsos negativos, la latencia requerida y el comportamiento por clase. Los tiempos reportados corresponden al entorno de ejecución del notebook y pueden variar según el hardware y la configuración.

## 6. Análisis por clase

El notebook reporta `mAP@50:95` por clase para cada modelo. Los resultados son:

| Modelo | chemical hazard | fire | no helmet | smoke | water leak |
|---|---:|---:|---:|---:|---:|
| YOLO11n | 0.2762 | 0.2708 | 0.4859 | 0.2669 | 0.3697 |
| YOLO11s | 0.2528 | 0.2358 | **0.4898** | 0.2411 | 0.3214 |
| YOLO26n | 0.2600 | **0.2831** | 0.4818 | 0.2607 | 0.3685 |
| YOLO26s | **0.3224** | 0.2609 | 0.4843 | 0.2610 | 0.3308 |

## 7. Experimento adicional: NMS vs. end-to-end

El notebook compara dos modos de inferencia para el mejor YOLO26 seleccionado por `mAP@50:95`:

| Modo | Precision | Recall | mAP@50 | mAP@50:95 | Inferencia (ms/imagen) |
|---|---:|---:|---:|---:|---:|
| Con NMS | 0.7478 | 0.5952 | 0.6415 | 0.3319 | 12.3421 |
| End-to-end sin NMS | 0.6818 | 0.5796 | 0.6008 | 0.3030 | 12.4449 |

En esta ejecución, el modo con NMS obtuvo mejores métricas y un tiempo de inferencia ligeramente menor. Esta observación se limita al modelo, dataset y entorno evaluados.

## 8. Estructura del repositorio

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── iaii-final-comparacion-justa-yolo11-yolo26.ipynb
├── data/
│   └── README.md
├── models/
│   └── README.md
├── results/
│   └── README.md
└── docs/
    └── README.md
```

- `notebooks/`: notebook principal con preparación de datos, entrenamiento y evaluación.
- `data/`: instrucciones para obtener y configurar el dataset. No contiene datos pesados.
- `models/`: espacio previsto para documentar checkpoints seleccionados, si se decide publicarlos.
- `results/`: tablas, gráficas y archivos CSV exportados por el notebook.
- `docs/`: documentación complementaria y material académico.

## 9. Requisitos y ejecución

### Entorno recomendado

El notebook fue preparado para ejecutarse en **Kaggle Notebooks** con Python, GPU e Internet habilitados. En la configuración de Kaggle:

1. Abrir el notebook desde el enlace de Kaggle o cargar la copia de `notebooks/`.
2. Habilitar acelerador GPU.
3. Habilitar Internet.
4. Agregar el dataset **Industrial Hazards Detection** como entrada.
5. Crear el secreto `WANDB_API_KEY` en Kaggle.
6. Verificar que `WANDB_ENTITY` y `WANDB_PROJECT` correspondan al espacio de trabajo del equipo.
7. Ejecutar las celdas en orden.

El notebook define `WANDB_ENTITY = "proyectoIAII"` y `WANDB_PROJECT = "industrial-hazard-detection"`. Si el identificador del equipo cambia, debe actualizarse `WANDB_ENTITY`.

### Instalación local (referencia)

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

La ejecución completa requiere adaptar las rutas del dataset, la configuración de GPU y el acceso a Weights & Biases. La integración con `kaggle_secrets` del notebook está diseñada para Kaggle.

## 10. Archivos de resultados

Al finalizar, el notebook exporta:

- `comparacion_final_test.csv`
- `resultados_por_clase_test.csv`
- `comparacion_nms_end_to_end.csv`

También genera los checkpoints `best.pt` de cada modelo y registra métricas y artefactos en Weights & Biases. Los archivos se guardan inicialmente en `/kaggle/working/`; deben descargarse o copiarse a `results/` o `models/` si se desea versionarlos.

## 11. Limitaciones y consideraciones

- Los resultados corresponden a una ejecución y configuración específicas; pueden variar con el hardware, versiones de librerías y condiciones de entrenamiento.
- La evaluación en el conjunto `test` es la referencia principal reportada en este trabajo.
- La distribución de clases no es uniforme; por ello se incluyen métricas por clase.
- Una evaluación académica no equivale a una certificación de seguridad industrial. Antes de un despliegue real se requieren validación en campo, análisis de riesgos y pruebas bajo condiciones operativas representativas.
- No se deben subir al repositorio secretos, tokens, credenciales, datasets con restricciones ni checkpoints grandes sin revisar las políticas aplicables.

## 12. Seguimiento de experimentos

Los experimentos se registran en **Weights & Biases**:

- Entidad configurada: `proyectoIAII`
- Proyecto: `industrial-hazard-detection`

El acceso a los paneles depende de los permisos del equipo. No se incluyen credenciales en este repositorio.

## 13. Equipo de trabajo

Completar esta sección antes de publicar el repositorio:

| Integrante | Rol / responsabilidad |
|---|---|
| Nombre del integrante 1 | [Responsabilidad] |
| Nombre del integrante 2 | [Responsabilidad] |
| Nombre del integrante 3 | [Responsabilidad] |

**Asignatura:** Inteligencia Artificial II  
**Institución:** [Nombre de la institución]  
**Docente:** [Nombre del docente]  
**Periodo académico:** [Periodo / año]

## 14. Cita académica

Si se utiliza este repositorio en informes, presentaciones o trabajos posteriores, citar el proyecto de acuerdo con los datos institucionales que complete el equipo. Se recomienda añadir autores, institución, año y URL definitiva del repositorio cuando esté publicado.

---

**Estado del documento:** versión inicial de documentación basada en el notebook final adjunto.  
