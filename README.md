# Segmentación de radiografías panorámicas dentales

Reimplementación en Python de la metodología de segmentación automática de piezas dentales en radiografías panorámicas, basada en procesamiento digital de imágenes clásico (sin redes neuronales). Es el punto de partida para explorar mejoras en el análisis de imágenes odontológicas.

> **Aviso:** este software es experimental, con fines académicos y de investigación. **No es un dispositivo médico y no debe usarse para diagnóstico ni para decisiones clínicas.**

## Referencia

Este proyecto implementa el método propuesto en:

> García Alcalá, L. M. (2020). *Segmentación de radiografías panorámicas dentales mediante técnicas de procesamiento digital de imágenes*. Tesis de Ingeniería en Sistemas Computacionales, Instituto Tecnológico de Estudios Superiores de Zamora. Asesores: Dr. Francisco Javier Cuevas de la Rosa (Centro de Investigaciones en Óptica, A.C.) y M.C. Juan Pablo Guerra Ibarra (ITESZ).

El método, los parámetros y las tablas de proporciones dentales son de esa tesis; todo el crédito del diseño del algoritmo corresponde a su autor y asesores. Este repositorio es una reimplementación independiente y **no** incluye el texto de la tesis. La tesis original usa C++; aquí se reescribió en Python. Los resultados pueden diferir de los reportados allí, y algunos detalles que la tesis deja ambiguos se resolvieron con supuestos propios (ver `docs/guia_funciones.md`).

## Fases

1. Preprocesamiento: filtro de la mediana y expansión del histograma.
2. Región de interés inicial: binarización, componentes conexas y proyección integral parabólica.
3. Separación de arcadas: regresión polinomial por mínimos cuadrados.
4. Curvas de diastemas.
5. Detección de espacios interdentales (Sobel + negativo, ventanas, filtro y centrado).
6. Aislamiento de las piezas dentales.

La explicación de cada función (pseudocódigo, intuición y matemática) está en [`docs/guia_funciones.md`](docs/guia_funciones.md).

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

## Uso

```bash
python main.py ruta/a/radiografia.jpg -o output   # una imagen (o una carpeta con .jpg)
python paso_a_paso.py ruta/a/radiografia.jpg      # muestra cada paso con matplotlib
python show_results.py <nombre>                   # overlay + piezas ordenadas desde output/
python generar_etapas.py                          # guarda mediana, histograma, binarización y región inicial de 10 imágenes de img/
python -m pytest tests
```

## Datos

Este repositorio **no incluye radiografías**: son datos de pacientes y no se publican. Para probar el código usa tus propias panorámicas dentales (con permiso y sin datos personales) o un conjunto de datos público. Coloca las imágenes en una carpeta `img/` (ignorada por git).

## Licencia

Copyright © 2026 PauAlba. Todos los derechos reservados. Puedes ver el código, pero no copiarlo, modificarlo ni redistribuirlo sin permiso.
