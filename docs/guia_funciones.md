# Guía de funciones — Segmentación de panorámicas dentales

Cómo ver el algoritmo en pantalla: `python paso_a_paso.py img/rad2.jpg` (una ventana por paso, 20 en total).

Notación: la imagen `I` tiene `M` filas (alto) y `N` columnas (ancho). `I(i,j)` = brillo del pixel en fila `i`, columna `j` (0 = negro, 255 = blanco). En la imagen, **y crece hacia abajo**.

## Mapa general

```
imagen ─► F1 limpiar ─► F2 hallar zona entre arcadas ─► F3 curva central
       ─► F4 dos curvas de diastemas ─► F5 puntos entre dientes ─► F6 recortar dientes
```

| Paso (`paso_a_paso.py`) | Fase | Archivo |
|---|---|---|
| 1–3 | F1 Preprocesamiento | `preprocess.py` |
| 4–9 | F2 Región de interés | `roi.py` |
| 10–13 | F3 Separar arcadas | `arch_split.py` |
| 14 | F4 Curvas de diastemas | `diastema_curves.py` |
| 15–18 | F5 Espacios interdentales | `interdental.py` |
| 19–20 | F6 Aislar piezas | `isolate.py` |

---

# Fase 1 · Preprocesamiento (`preprocess.py`)

## `preprocess(gray, ksize=3)`
**Primaria:** primero quitamos las "motitas" de la foto (ruido) y luego la estiramos para que los blancos sean bien blancos y los negros bien negros.

```
preprocess(gray):
    suave = mediana(gray, 3x3)
    return expandir_histograma(suave)
```
**Mate:** composición `G = E(Med(I))`.

**Símbolos:**
- `I`: imagen original. `Med(·)`: aplicar el filtro de la mediana. `E(·)`: aplicar la expansión del histograma. `G`: imagen final lista para la fase 2.
- Se lee de adentro hacia afuera: primero `Med`, luego `E`.

## `cv2.medianBlur` (filtro de la mediana 3×3, se llama dentro de `preprocess`)
**Primaria:** para cada pixel miras a sus 8 vecinos, ordenas los 9 números de menor a mayor y te quedas con el del medio. Un pixel "loco" (muy brillante o muy oscuro) nunca es el del medio, así que desaparece.

```
para cada pixel (i,j):
    vecinos = valores de la ventana 3x3 centrada en (i,j)
    salida(i,j) = mediana(vecinos)
```
**Mate:** `S(i,j) = mediana{ I(i+u, j+v) : u,v ∈ {-1,0,1} }`. Ejemplo: {10, 12, 11, 255, 13, 9, 12, 10, 11} → ordenado, el 5.º es 11.

**Símbolos:**
- `S(i,j)`: brillo del pixel (fila `i`, columna `j`) en la imagen suavizada (salida).
- `I(i+u, j+v)`: brillo del vecino que está `u` filas y `v` columnas desplazado del pixel central.
- `u, v ∈ {-1, 0, 1}`: `u` y `v` toman los valores −1 (uno antes), 0 (el mismo) o 1 (uno después). Combinados dan los 9 pixeles de la ventana 3×3.
- `mediana{…}`: ordenar esos 9 números y tomar el 5.º (el del centro).
- `{ … : … }`: notación de conjunto: "el conjunto de los valores de la izquierda, tales que se cumple lo de la derecha".

## `stretch_histogram(img)`
**Primaria:** si la foto es "grisácea" (el más oscuro es 50 y el más claro 200), la estiras como liga para que el más oscuro valga 0 y el más claro 255.

```
lo = mínimo(img); hi = máximo(img)
si (lo == 0 y hi == 255) o hi == lo: devolver img sin cambios
para cada pixel: nuevo = (pixel - lo) * 255 / (hi - lo)
redondear y recortar a [0,255]
```
**Mate (Ec. 2.25):** `G(i,j) = 255 · (I(i,j) − I_min) / (I_max − I_min)`. Es una recta que manda `I_min→0` e `I_max→255`. Si ya son 0 y 255, `G = I` (Ec. 3.1) y no se hace nada.

**Símbolos:**
- `I(i,j)`: brillo original del pixel. `G(i,j)`: brillo nuevo.
- `I_min`, `I_max`: el brillo más oscuro y el más claro que existen en toda la imagen.
- `I − I_min`: corre la escala para que el más oscuro valga 0. `/ (I_max − I_min)`: lo convierte en una fracción entre 0 y 1. `255 ·`: la devuelve a la escala 0–255.
- Ejemplo: `I_min=50`, `I_max=200`, pixel=125 → `255·(125−50)/150 = 127.5` (queda al centro, como debe).

---

# Fase 2 · Región de interés inicial (`roi.py`)

Objetivo: encontrar el hueco oscuro que queda **entre los dientes de arriba y los de abajo**.

## `initial_rect(shape, p)`
**Primaria:** dibujas un rectángulo en el centro de la foto, porque ahí suele estar la boca cerrada entre las dos filas de dientes.

```
h = 0.2 * M ; w = 0.6 * N
fila_inicial = (M - h) / 2 ; col_inicial = (N - w) / 2
devolver rectángulo(fila_inicial, fila_inicial+h, col_inicial, col_inicial+w)
```
**Mate:** `E = [r_i, r_f] × [c_i, c_f]` con `r_f−r_i = 0.2M`, `c_f−c_i = 0.6N`, centrado: `r_i = (M−0.2M)/2 = 0.4M`.

**Símbolos:**
- `E`: el rectángulo de búsqueda. `×`: el rectángulo es el producto de un intervalo de filas por un intervalo de columnas.
- `[r_i, r_f]`: filas desde la inicial (`i`) hasta la final (`f`). `[c_i, c_f]`: columnas, igual.
- `M`, `N`: alto y ancho de la imagen en pixeles. `0.2M` = 20 % del alto, `0.6N` = 60 % del ancho.
- `(M−0.2M)/2`: el espacio sobrante (arriba + abajo) repartido en dos mitades, para dejar el rectángulo centrado. Da `0.4M`: empieza al 40 % de la altura (y termina al 60 %).

## `threshold_from_region(img, rect)`
**Primaria:** miramos qué tan oscuro es el rectángulo. Si es, digamos, 40 % de oscuro, decidimos que "oscuro" son los pixeles que están dentro del 40 % más oscuro de toda la foto. Ese corte se llama umbral θ.

```
media = promedio de pixeles dentro del rectángulo
P = 100 * media / 255              # de 0 a 100
hist = cuántos pixeles hay de cada brillo 0..255
acumulado[k] = % de pixeles con brillo ≤ k
θ = 0
mientras θ < 255 y acumulado[θ] < P: θ = θ + 1
devolver θ
```
**Mate (Ecs. 3.2, 3.3):**
`Ī = (1/S) Σ_{i=r_i}^{r_f} Σ_{j=c_i}^{c_f} I(i,j)`, `P = 100·Ī/255`,
`θ = min { k : F(k) ≥ P }` donde `F(k) = 100 · (#pixeles ≤ k) / (M·N)` es la distribución acumulada.

**Símbolos:**
- `Ī` (se lee "I barra"): brillo promedio del rectángulo. `S`: cuántos pixeles tiene el rectángulo.
- `Σ` (sigma): "sumar". `Σ_{i=r_i}^{r_f}` significa sumar variando `i` desde `r_i` hasta `r_f`; el segundo `Σ` hace lo mismo con las columnas `j`. Entre ambos suman todos los pixeles del rectángulo.
- `P`: el promedio convertido a porcentaje (0 % = negro total, 100 % = blanco total). Se multiplica por 100 y se divide entre 255.
- `F(k)`: porcentaje de pixeles de toda la imagen con brillo menor o igual a `k`. `#pixeles ≤ k` = "cuántos pixeles valen k o menos". `M·N` = total de pixeles.
- `θ` (theta): el umbral. `min{ k : F(k) ≥ P }` = "el menor brillo `k` para el cual ya se acumuló al menos `P` % de la imagen".

## `binarize(img, theta)`
**Primaria:** pintas de blanco todo lo oscuro (≤ θ) y de negro todo lo claro. Queda un dibujo de dos colores.

```
para cada pixel: salida = 1 si pixel <= θ, si no 0
```
**Mate:** `B(i,j) = 1 si I(i,j) ≤ θ; 0 en otro caso`.

**Símbolos:**
- `B(i,j)`: pixel de la imagen binaria (solo puede valer 0 = negro o 1 = blanco).
- `I(i,j)`: brillo del pixel original. `θ`: umbral calculado antes.
- `≤`: "menor o igual". Es una regla de decisión: oscuro (≤ θ) → 1; claro (> θ) → 0.

## `largest_component(mask)`
**Primaria:** en el dibujo blanco/negro hay "islas" blancas. Contamos cuántos pixeles tiene cada isla y nos quedamos con la más grande. Es como pintar con el bote de pintura (FloodFill): donde cae la pintura se llena toda la isla.

```
etiquetar cada isla de pixeles blancos conectados (vecinos arriba/abajo/izq/der)
área[k] = número de pixeles de la isla k
devolver máscara de la isla con mayor área, y su área
```
**Mate:** componentes conexas de `B` con conectividad 4: dos pixeles blancos están en la misma componente si hay un camino de pixeles blancos adyacentes entre ellos. Se elige `k* = argmax_k |C_k|`.

**Símbolos:**
- `B`: imagen binaria. `C_k`: la isla número `k`. `|C_k|`: su tamaño (cuántos pixeles tiene).
- Conectividad 4: se consideran vecinos solo arriba, abajo, izquierda y derecha (no las diagonales).
- `argmax_k |C_k|`: "el valor de `k` que hace máximo `|C_k|`". Es decir, el número de la isla más grande (no su tamaño). `k*` es el ganador.

## `parabola_through(p0, p1, p2)`
**Primaria:** con 3 puntos que no están en línea recta, existe una única "sonrisa" (parábola) que pasa por los tres. Resolvemos un acertijo de ecuaciones para encontrarla.

```
armar matriz A con filas [x², x, 1] para cada punto
armar vector y con las alturas
resolver A · (a,b,c) = y
devolver (a,b,c)
```
**Mate (Ec. 3.5):** `f(x) = ax² + bx + c`, y se cumple
```
⎡x0² x0 1⎤⎡a⎤   ⎡y0⎤
⎢x1² x1 1⎥⎢b⎥ = ⎢y1⎥
⎣x2² x2 1⎦⎣c⎦   ⎣y2⎦
```
Tiene solución única si los `x` son distintos y los puntos no están alineados (por eso el punto medio se desplaza `sag ≠ 0`).

**Símbolos:**
- `f(x) = ax² + bx + c`: la parábola. `x` es la posición horizontal; `f(x)` la altura. `a, b, c` son las tres incógnitas: `a` = qué tan cerrada, `b` = inclinación, `c` = altura base.
- `(x0,y0), (x1,y1), (x2,y2)`: los tres puntos por los que debe pasar.
- La matriz: cada fila sale de sustituir un punto en `f`: `a·x² + b·x + c = y`. Tres puntos → tres ecuaciones → se resuelve y salen `a, b, c`.
- Por qué "no alineados": si los tres puntos estuvieran en una recta, ninguna parábola "de verdad" pasaría por ellos y el sistema no tendría solución única.

## `curve_sum(img, xs, ys)`
**Primaria:** caminas por encima de una curva dibujada en la foto y promedias el brillo de lo que pisas. Una curva que pisa zonas oscuras da un número chico.

```
para cada x en xs: tomar pixel (fila=round(ys[x]), col=x)
devolver el promedio de esos pixeles
```
**Mate (proyección integral):** `S(f) = (1/|X|) Σ_{x∈X} I(f(x), x)`. Generaliza la proyección por fila (`f` constante) a una curva `f` cualquiera.

**Símbolos:**
- `S(f)`: el "puntaje" de la curva `f`; entre más bajo, más oscura la ruta.
- `X`: el conjunto de columnas que recorre la curva. `|X|`: cuántas son. `1/|X|`: dividir entre ese número, o sea, promediar.
- `x∈X`: "para cada `x` que pertenece a `X`". `f(x)`: la fila por la que pasa la curva en la columna `x`.
- `I(f(x), x)`: brillo del pixel donde cae la curva en esa columna.

## `best_parabola(img, rect, row_c, p)`
**Primaria:** probamos muchas sonrisas (parábolas) más arriba, más abajo, más curvas, menos curvas, y nos quedamos con la que camina por lo más oscuro: ahí está el hueco entre los dientes.

```
mejor = ∞
para shift en -40..40 paso 10:
  para sag en -60..60 paso 10 (sag ≠ 0):
      r = row_c + shift
      parábola por (izq, r), (medio, r + sag), (der, r)
      s = curve_sum(img, parábola)
      si s < mejor: guardar parábola
devolver la mejor
```
**Mate:** `f* = argmin_{f∈F} S(f)`, con la familia `F` generada variando `shift` y `sag`. Ambos pasan por los extremos del rectángulo a altura `r`, y por el medio a `r+sag`.

**Símbolos:**
- `F`: el conjunto de todas las parábolas candidatas (en total 9 valores de `shift` × 12 de `sag` ≈ 108).
- `argmin_{f∈F} S(f)`: "la `f` dentro de `F` que hace mínimo `S(f)`". `f*` es la ganadora (la que pisa lo más oscuro).
- `shift`: cuánto se sube o baja toda la parábola. `sag`: cuánto se hunde (o se levanta) el punto medio respecto a los extremos; da la curvatura.
- `r`: fila de referencia de los dos extremos.

## `detect_roi(img, p)` — junta todo F2
**Primaria:** (1) rectángulo central, (2) sacar el umbral y pintar blanco/negro, (3) deslizar el rectángulo un poquito arriba y abajo y quedarnos con la posición donde cabe la isla blanca más grande, (4) afinar con la mejor parábola y centrar el rectángulo ahí.

```
rect0 = initial_rect
θ = threshold_from_region ; B = binarize
mejor_dy = argmax_{dy ∈ [-αM, αM]} área( largest_component( B[rect0 desplazado dy] ) )
rect1 = rect0 desplazado mejor_dy
f = best_parabola(rect1)
fila_centro = promedio de f(x) en el rectángulo
rect = rect1 re-centrado verticalmente en fila_centro
devolver rect, B, θ
```
**Mate:** `dy* = argmax_{|dy| ≤ αM} |C_max(B ∩ (E + dy))|` (Ec. 3.4, α = 0.05). Luego `R_nuevo = mean_x f*(x)`.

**Símbolos:**
- `dy`: cuántos pixeles se desplaza el rectángulo hacia arriba (−) o abajo (+). `|dy| ≤ αM`: no más de `α·M` pixeles (con `α = 0.05`, el 5 % del alto).
- `E + dy`: el rectángulo `E` movido `dy` filas. `B ∩ (E + dy)`: la parte de la imagen binaria que queda dentro de él (`∩` = "intersección", lo que tienen en común).
- `C_max(·)`: la componente conexa más grande de esa porción. `|…|`: su número de pixeles.
- `dy*`: el desplazamiento ganador.
- `mean_x f*(x)`: promedio (media) de la altura de la parábola ganadora en todas las columnas `x`; `R_nuevo` es la fila donde se re-centra el rectángulo.

---

# Fase 3 · Separar las arcadas (`arch_split.py`)

## `median_filter_1d(v, k)`
**Primaria:** igual que la mediana de antes pero en una fila de números: cada número se reemplaza por el del medio entre él y sus 12 vecinos de cada lado. Quita los "picos".

```
k = k impar ; copiar bordes (k//2 veces)
para cada posición i: salida[i] = mediana(v[i-k//2 .. i+k//2])
```
**Mate:** `v'_i = mediana{ v_{i−m}, …, v_{i+m} }`, con `m = (K−1)/2` (K = 25).

**Símbolos:**
- `v_i`: el `i`-ésimo número de la lista original; `v'_i`: el mismo ya filtrado.
- `K`: tamaño de la ventana (25 números). `m`: cuántos vecinos hay a cada lado (`(25−1)/2 = 12`).
- `v_{i−m}, …, v_{i+m}`: el número `i` más sus 12 vecinos de la izquierda y 12 de la derecha. Se toma la mediana de esos 25.
- Los puntos `…` significan "y así sucesivamente, todos los intermedios".

## `characteristic_points(binary, rect)`
**Primaria:** tomamos la isla oscura entre las arcadas. Le "recortamos" las partes que sobresalen por arriba y, para cada columna, marcamos el punto del medio de la isla. Esos puntos forman una línea punteada que sigue la boca.

```
sub = isla más grande de B dentro de rect
para cada columna c con pixeles blancos:
    primero[c], último[c] = primer y último blanco
altura_media = promedio(último - primero + 1)
primero[c] = max(primero[c], último[c] - altura_media + 1)   # recorta lo que sobresale arriba
punto[c] = ((primero[c] + último[c]) / 2) + fila_inicial_del_rect
devolver (x = columna, y = punto)
```
**Mate:** `h_c = u_c − f_c + 1`, `h̄ = mean(h_c)`; nuevo `f_c = max(f_c, u_c − h̄ + 1)`; `y_c = (f_c + u_c)/2`. Resume cientos de pixeles por columna en un solo punto `(c, y_c)`.

**Símbolos:**
- `c`: número de columna. `f_c` (first): fila del primer pixel blanco en esa columna. `u_c`: fila del último pixel blanco.
- `h_c = u_c − f_c + 1`: altura de la isla en esa columna (se suma 1 porque se cuentan ambos extremos). `h̄ = mean(h_c)`: altura promedio de todas las columnas.
- `max(f_c, u_c − h̄ + 1)`: se queda el mayor de los dos números (más abajo en la imagen); así se recorta lo que sobresale por arriba de la altura promedio.
- `y_c = (f_c + u_c)/2`: el punto medio entre el techo y el piso de la isla.

## `fit_polynomial(xs, ys, degree, k)`
**Primaria:** primero suavizamos los puntos (quitamos picos) y después buscamos la curva suave que pasa "lo más cerca posible" de todos. Es como estirar un hilo entre una nube de puntos.

```
ys' = median_filter_1d(ys, 25)
devolver polinomio de grado 2 que minimiza Σ (polinomio(x) - ys')²
```
**Mate (mínimos cuadrados):** se busca `a = (a_0..a_d)` que minimiza `J(a) = Σ_i (y_i − Σ_k a_k x_i^k)²`. Derivando e igualando a cero salen las **ecuaciones normales** `(XᵀX) a = Xᵀy`, un sistema `(d+1)×(d+1)` (3×3 para grado 2) donde `X` es la matriz de Vandermonde. `np.polyfit` lo resuelve.

**Símbolos:**
- `(x_i, y_i)`: los puntos de datos (columna, altura). `i` recorre todos.
- `a = (a_0, …, a_d)`: los coeficientes del polinomio; `d` es su grado (2 → parábola). `Σ_k a_k x^k = a_0 + a_1x + a_2x² + …` es el polinomio evaluado en `x`.
- `y_i − Σ_k a_k x_i^k`: error en el punto `i` (dato real menos lo que predice la curva). Se eleva al cuadrado para que sea siempre positivo.
- `J(a)`: error total. Buscamos los `a` que lo hacen lo más pequeño posible.
- `X`: matriz de Vandermonde (cada fila es `[1, x_i, x_i², …]`). `Xᵀ`: su transpuesta (filas ↔ columnas). `y`: vector de alturas. `(XᵀX) a = Xᵀy`: sistema de ecuaciones que da directamente los `a` óptimos.

## `split_arches(binary, rect, p)`
**Primaria:** une los dos pasos anteriores: saca los puntos y dibuja la curva. Esa curva es la "línea de la sonrisa" que separa dientes de arriba y de abajo.

```
(xs, ys) = characteristic_points
devolver fit_polynomial(xs, ys, grado=2, K=25)
```
**Mate:** `C(x) = a·x² + b·x + c`.

**Símbolos:**
- `C(x)`: la altura (fila) de la curva central en la columna `x`. `a, b, c`: los coeficientes que encontró el ajuste (ver `fit_polynomial`).

---

# Fase 4 · Curvas de diastemas (`diastema_curves.py`)

## `_shifted_mean(img, coef, x0, x1, dy)` (auxiliar)
**Primaria:** copia la curva central, muévela `dy` pixeles hacia arriba/abajo y calcula el brillo promedio por donde pasa.
```
ys = C(x) + dy para x en [x0, x1)
si la curva se sale de la imagen: devolver ∞
devolver promedio de img(ys, xs)
```
**Mate:** `S(dy) = (1/n) Σ_x I(C(x)+dy, x)`.

**Símbolos:**
- `dy`: desplazamiento vertical (positivo = hacia abajo). `C(x)+dy`: la curva movida `dy` filas.
- `n`: número de columnas recorridas; `(1/n) Σ_x`: promedio sobre todas ellas.
- `I(·, x)`: brillo del pixel de esa columna en la fila indicada. `S(dy)`: brillo promedio de la ruta desplazada.

## `diastema_curves(img, coef, rect, p)`
**Primaria:** los huequitos entre dientes son oscuros. Subimos la curva central entre 35 % y 45 % de la altura del rectángulo y escogemos la altura donde la curva es lo más oscura posible: ahí corre una línea de huequitos entre los dientes de arriba. Hacemos lo mismo hacia abajo para la mandíbula.

```
H = altura del rect
para d en [0.35H .. 0.45H]:
    arriba_d = promedio de brillo con la curva desplazada -d
d_s = el d con menor promedio ; igual con +d para d_i
curva_superior = C(x) - d_s ; curva_inferior = C(x) + d_i
```
**Mate (Ecs. 3.6, 3.7):** `d_s = argmin_{d∈[0.35H,0.45H]} S(−d)`, `d_i = argmin_{d∈[0.35H,0.45H]} S(+d)`. Desplazar verticalmente un polinomio solo cambia su término independiente: `c' = c ∓ d`.

**Símbolos:**
- `H`: altura del rectángulo de la fase 2. `0.35H`–`0.45H`: rango de distancias donde se busca (valores estadísticos de la tesis).
- `d`: distancia vertical probada. `d_s`: la mejor distancia hacia arriba (superior, *s*); `d_i`: la mejor hacia abajo (inferior, *i*).
- `S(−d)`: puntaje con la curva subida `d` pixeles (en la imagen "arriba" es y menor, por eso el signo menos); `S(+d)`: bajada.
- `argmin`: el valor de `d` que da el puntaje más bajo (la ruta más oscura).
- `c' = c ∓ d`: el nuevo término independiente; `−` para la curva superior, `+` para la inferior. `a` y `b` no cambian, por eso la forma es idéntica.

---

# Fase 5 · Espacios interdentales (`interdental.py`)

## `auxiliary_image(img)`
**Primaria:** hacemos que los huequitos entre dientes se vean brillantes de dos maneras (invirtiendo la foto como un negativo, y marcando los bordes) y sumamos ambas. Así son más fáciles de encontrar.

```
gx, gy = derivadas de Sobel en x y en y
borde = normalizar_0_255( sqrt(gx² + gy²) )
negativo = 255 - img
devolver (negativo + borde) / 2
```
**Mate (Ec. 2.32):** `|∇I| = √(Gx² + Gy²)`, con Sobel `Gx = [[-1,0,1],[-2,0,2],[-1,0,1]] * I` y `Gy` su transpuesta. `A = ((255 − I) + |∇I|_norm) / 2`.

**Símbolos:**
- `∇I` (nabla I): el gradiente, un par `(Gx, Gy)` que dice cuánto cambia el brillo hacia la derecha (`Gx`) y hacia abajo (`Gy`). `|∇I|`: su longitud (qué tan fuerte es el borde), por Pitágoras `√(Gx² + Gy²)`.
- `*`: convolución: deslizar la máscara 3×3 sobre la imagen, multiplicar cada vecino por su número de la máscara y sumar. `Gx` detecta bordes verticales (izquierda oscuro/derecha claro); `Gy` los horizontales (es la transpuesta de `Gx`).
- `|∇I|_norm`: la magnitud reescalada a 0–255.
- `255 − I`: negativo. `A`: imagen auxiliar, el promedio de ambos (se divide entre 2 para no pasar de 255).

## Clase `Arch` (curva muestreada)
**Primaria:** una regla flexible: puedes preguntar "¿cuál es el punto a 100 pixeles de camino desde el principio de la curva?".
```
xs = x0..x1 ; ys = polinomio(xs)
s[i] = distancia recorrida a lo largo de la curva hasta el punto i
length = s[último]
point_at(s) = interpolar (x,y) en la distancia s
idx_at(s)   = índice del punto más cercano a la distancia s
```
**Mate (longitud de arco):** `s_i = Σ_{k<i} √(Δx_k² + Δy_k²)`. Se usa arco y no `x` porque las piezas miden lo mismo *a lo largo* de la curva.

**Símbolos:**
- `s_i`: distancia recorrida desde el inicio de la curva hasta el punto `i`. `Σ_{k<i}`: sumar los segmentos desde el primero hasta el anterior al `i`.
- `Δx_k`, `Δy_k`: cuánto avanza en horizontal y en vertical el segmento `k` (diferencia entre dos puntos consecutivos).
- `√(Δx² + Δy²)`: longitud de ese segmento (Pitágoras). Sumando muchos segmentos pequeños se obtiene la longitud de la curva (`length = L`).

## `_window_scores(aux, arch, p)` (auxiliar)
**Primaria:** en cada punto de la curva pegamos una ventanita de 31×9 pixeles y sumamos su brillo. Donde la suma es grande, probablemente hay un huequito entre dientes.
```
integral = tabla de sumas acumuladas de aux
para cada punto (x,y) de la curva:
    si la ventana cabe en la imagen:
        score = suma de aux dentro de ventana 31 (alto) x 9 (ancho) centrada en (x,y)
    si no: -∞
```
**Mate:** con la **imagen integral** `Σ(r,c) = Σ_{i<r,j<c} A(i,j)`, la suma de cualquier rectángulo se obtiene con 4 lecturas: `Σ(r1,c1) − Σ(r0,c1) − Σ(r1,c0) + Σ(r0,c0)`. Costo O(1) por ventana.

**Símbolos:**
- `Σ(r,c)` (aquí es la tabla, no una suma suelta): suma de todos los pixeles de `A` con fila `< r` y columna `< c`, o sea el rectángulo de la esquina superior izquierda hasta `(r,c)`.
- `A(i,j)`: brillo de la imagen auxiliar. `r0, r1`: filas superior e inferior de la ventana; `c0, c1`: columnas izquierda y derecha.
- Los cuatro términos: el rectángulo grande `(r1,c1)`, menos lo que sobra arriba `(r0,c1)`, menos lo que sobra a la izquierda `(r1,c0)`, más la esquina `(r0,c0)` que se restó dos veces.
- `O(1)`: el tiempo no depende del tamaño de la ventana.

## `detect_diastemas(aux, coef, x0, x1, widths_rel, p)`
**Primaria:** sabemos más o menos cuánto mide cada diente (Tabla 3.1: un molar es más ancho que un incisivo). Con eso adivinamos dónde deberían estar las 17 "rayitas" entre dientes y buscamos, cerca de cada una, el lugar más brillante. Después borramos las rayitas demasiado juntas (quedándonos con la más brillante) y finalmente movemos cada una un poquito hasta su mejor lugar.

```
arch = Arch(curva, x0, x1) ; sc = _window_scores
px[k] = widths_rel[k] * arch.length             # ancho esperado de cada pieza en pixeles
límites[k] = suma acumulada de px                # 17 posiciones esperadas

# 1) Búsqueda por intervalo
para cada límite k:
    mitad = 0.5 * min(px de las piezas vecinas)
    idx[k] = argmax de sc en [límite - mitad, límite + mitad]

# 2) Filtro de distancia euclidiana
ancho_prom = promedio(px)
recorrer idx en orden:
    si distancia al anterior conservado < 0.7 * ancho_prom:
        quedarse con el de mayor sc
    si no: conservarlo

# 3) Centrado local
para cada punto conservado:
    rango = 0.3 * ancho_prom, sin pasar de la mitad hacia sus vecinos
    mover al argmax de sc dentro del rango
devolver lista de (x, y)
```
**Mate:**
- Anchos esperados (Ec. 3.8): `A_i = (P_i/100) · L`, donde `P_i` viene de la Tabla 3.1 normalizada para que `ΣP_i = 100 %` en las 16 piezas, y `L` es la longitud de arco de la arcada.
- Límites esperados: `b_k = Σ_{i<k} A_i`, `k = 0..16`.
- Búsqueda: `s_k* = argmax_{|s−b_k|≤h_k} W(s)`, `h_k = 0.5·min(A_{k−1}, A_k)`.
- Filtro: si `‖p_i − p_j‖ < 0.7·Ā`, se conserva `argmax(W(p_i), W(p_j))`.

**Símbolos:**
- `P_i`: porcentaje del ancho de la pieza `i` (Tabla 3.1). `L`: longitud de arco de la arcada en pixeles. `A_i`: ancho esperado de la pieza `i` en pixeles. `Ā`: promedio de todos los `A_i`.
- `b_k`: posición (medida sobre la curva) donde se espera el diastema `k`. `Σ_{i<k} A_i`: sumar los anchos de todas las piezas anteriores. Hay 16 piezas → 17 límites (`k = 0` a `16`).
- `W(s)`: puntaje de la ventana 31×9 en la posición `s` de la curva (suma de brillo de la imagen auxiliar).
- `h_k`: mitad del intervalo de búsqueda alrededor de `b_k`: la mitad de la más angosta de las dos piezas vecinas (`min(A_{k−1}, A_k)`).
- `|s − b_k| ≤ h_k`: posiciones a lo más a `h_k` de la esperada. `s_k*`: la posición con mayor `W`.
- `‖p_i − p_j‖`: distancia euclidiana entre dos puntos candidatos. `0.7·Ā`: si están más cerca que el 70 % de un diente promedio, se descarta el menos brillante.

## `detect_both(img, up_coef, low_coef, x0, x1, p)`
**Primaria:** hace la misma búsqueda para los dientes de arriba (con las proporciones de la Tabla 3.1 "superior") y los de abajo (con "inferior"), usando la imagen auxiliar.
```
aux = auxiliary_image(img)
up  = detect_diastemas(aux, curva_sup, ..., arch_widths(SUPERIOR))
low = detect_diastemas(aux, curva_inf, ..., arch_widths(INFERIOR))
```
**Mate:** `arch_widths` toma las 8 piezas de media arcada, las refleja (simetría izq./der.) y normaliza: `w_k = p_k / Σp`.

**Símbolos:**
- `p_k`: porcentaje de la pieza `k` (Tabla 3.1). `Σp`: suma de los 16 porcentajes (ya reflejados).
- `w_k`: fracción de la arcada que ocupa la pieza `k`; todas suman 1. Dividir entre la suma hace que no importe si los porcentajes de la tabla suman 93 % u 88 %.

---

# Fase 6 · Aislar las piezas (`isolate.py`)

## `isolate_arch(img, diastemas, coef, arch, p)`
**Primaria:** entre cada dos rayitas (diastemas) hay un diente. El ancho del diente es la distancia entre rayitas; el largo es 3 veces el ancho (los dientes son más o menos 3 veces más largos que anchos). Se dibuja un rectángulo en medio, se gira para que siga la curva de la boca y se recorta como con tijeras.

```
puntos = diastemas ordenados
anchos[i] = distancia(puntos[i], puntos[i+1])
largo = promedio(3 * anchos)                      # un solo largo por arcada
para cada par consecutivo:
    centro = punto medio entre los dos diastemas
    α = arctan( derivada de la curva en x_centro )     # inclinación
    girar la imagen α grados alrededor del centro
    recortar rectángulo (ancho_i × largo) centrado
    guardar Tooth(arco, índice, centro, ancho, largo, α, recorte)
```
**Mate:**
- Ancho (Ec. 3.9): `w_i = ‖x_i − x_{i+1}‖`.
- Largo (Ecs. 3.10–3.12): `r_i = l_i / w_i ∈ [2.16, 2.81]` (Tabla 3.2) ⇒ se aproxima `l ≈ 3w`; se usa el promedio `l̄ = mean(3 w_i)` para evitar largos exagerados.
- Orientación: la tangente a la curva en `x_c` tiene pendiente `f'(x_c) = 2a·x_c + b`; el ángulo de giro es `α = arctan(f'(x_c))` y la pieza (perpendicular a la curva) queda recta al girar la imagen `α`.
- Giro + recorte con una sola matriz afín `M = T · R(α, centro)`, donde `R` es la rotación `[[cos α, sin α],[−sin α, cos α]]` y `T` traslada el centro al medio del recorte.

**Símbolos:**
- `x_i`, `x_{i+1}`: dos diastemas consecutivos (puntos con coordenadas `(x, y)`). `‖·‖`: distancia euclidiana, `√(Δx² + Δy²)`. `w_i`: ancho de la pieza `i`.
- `l_i`: largo de la pieza; `r_i = l_i / w_i`: cuántas veces es más larga que ancha (entre 2.16 y 2.81 según la Tabla 3.2). Se usa `l ≈ 3w`, `l̄`: promedio de los largos de la arcada.
- `f'(x_c)`: derivada (pendiente) de la curva en la columna del centro `x_c`; para `ax²+bx+c` es `2a·x_c + b`.
- `α` (alfa): ángulo de inclinación. `arctan`: convierte una pendiente en ángulo (pendiente 1 → 45°).
- `R(α, centro)`: rotación de `α` grados alrededor de `centro`; `cos` y `sin` son coseno y seno de `α`. `T`: traslación (mover). `M = T · R`: aplicar primero `R`, luego `T`, en una sola operación (matriz afín).

---

# Funciones de apoyo

| Función | Archivo | Qué hace |
|---|---|---|
| `Params` | `config.py` | Todos los números ajustables (0.2, 0.6, α, K=25, 31×9, 70 %, etc.). |
| `arch_widths(pcts)` | `config.py` | Convierte la Tabla 3.1 (8 piezas) en 16 anchos relativos que suman 1. |
| `segment(gray)` | `pipeline.py` | Ejecuta F1→F6 en orden y devuelve un `SegmentationResult`. |
| `load_gray(path)` | `pipeline.py` | Lee la imagen en grises (soporta rutas con acentos/espacios). |
| `draw_overlay(gray, r)` | `pipeline.py` | Dibuja rectángulo, curvas, diastemas y cajas sobre la imagen. |
| `main.py` | raíz | Línea de comandos: procesa una imagen o carpeta y guarda en `output/`. |
| `show_results.py` | raíz | Muestra el overlay y los dientes ordenados (con número FDI). |
| `paso_a_paso.py` | raíz | Muestra cada paso con título. |

# Repaso rápido: una frase por fase

1. **F1:** limpiar (mediana) y estirar contraste.
2. **F2:** pintar blanco/negro, buscar la isla oscura más grande en el centro y afinarla con parábolas.
3. **F3:** puntos medios de esa isla → suavizar → curva por mínimos cuadrados.
4. **F4:** copiar la curva hacia arriba y hacia abajo hasta donde es más oscura.
5. **F5:** resaltar huequitos (negativo + Sobel), buscar los 17 más brillantes guiados por la Tabla 3.1, limpiar y centrar.
6. **F6:** entre cada par de huequitos, rectángulo (ancho = distancia, largo = 3·ancho), girado con la curva, recortado.

---

# Glosario matemático (qué significa cada herramienta)

| Concepto | Explicación breve | Dónde se usa |
|---|---|---|
| **Mediana** | De una lista ordenada, el valor del centro. A diferencia del promedio, un valor extremo no la mueve: en {10, 11, 12, 13, 255} el promedio es 60 pero la mediana es 12. | Filtro 3×3 (F1) y filtro 1D con K=25 (F3) |
| **Histograma** | Una tabla que cuenta cuántos pixeles hay de cada brillo (0 a 255). Muestra si la foto es oscura, clara o "grisácea". | Expansión del histograma (F1), umbral (F2) |
| **Histograma acumulado** `F(k)` | El % de pixeles con brillo ≤ k. Sube de 0 % a 100 %. Sirve para decir "el 40 % más oscuro de la foto". | Cálculo de θ (F2) |
| **Umbral / binarización** | Una línea de corte: lo que está de un lado se vuelve 1 y del otro 0. | `binarize` (F2) |
| **Componente conexa** | Un grupo de pixeles blancos que se tocan entre sí (una "isla"). Se cuentan con FloodFill (el bote de pintura). | `largest_component` (F2, F3) |
| **Parábola** `f(x)=ax²+bx+c` | Curva en forma de sonrisa (`a>0`) o de arco (`a<0`). `a` controla qué tan cerrada es, `b` la inclina y `c` la sube o baja. Sumarle un número a `c` la desplaza verticalmente sin cambiar su forma. | Regiones entre arcadas y curvas de diastemas (F2–F4) |
| **Sistema de ecuaciones lineales** | Varias ecuaciones con las mismas incógnitas. Con 3 puntos se arma un sistema 3×3 cuya solución son `a, b, c`. Falla si los puntos están alineados (matriz singular). | `parabola_through` (F2) |
| **Proyección integral** | Sumar (o promediar) el brillo a lo largo de una línea. Aquí se generaliza a una curva: la curva con menor suma pasa por lo más oscuro. | `curve_sum`, `best_parabola`, `diastema_curves` |
| **Mínimos cuadrados** | Para ajustar una curva a muchos puntos, se elige la que hace más pequeña la suma de los errores al cuadrado. Elevar al cuadrado evita que errores + y − se cancelen y castiga más los errores grandes. | `fit_polynomial` (F3) |
| **Regresión polinomial** | Ajustar un polinomio (grado 2 = parábola, grado 4 = más ondulado) a datos. Más grado = más flexible pero puede oscilar. | F3 |
| **Derivada** `f'(x)` | La pendiente de la curva en un punto: qué tanto sube o baja. Para `ax²+bx+c` es `2ax+b`. | Orientación de las piezas (F6) |
| **Arco tangente** `arctan` | Convierte una pendiente en un ángulo: `arctan(1) = 45°`, `arctan(0) = 0°`. | Ángulo de giro `α` (F6) |
| **Gradiente / Sobel** | El gradiente mide qué tan rápido cambia el brillo; es grande en los bordes. Sobel lo estima con dos máscaras 3×3 (`Gx` horizontal, `Gy` vertical) y la magnitud es `√(Gx²+Gy²)` (teorema de Pitágoras). | `auxiliary_image` (F5) |
| **Negativo** `255 − I` | Invierte el brillo: lo oscuro se vuelve claro. Los diastemas oscuros pasan a ser brillantes. | `auxiliary_image` (F5) |
| **Imagen integral** | Tabla donde cada celda guarda la suma de todo lo que está arriba y a la izquierda. Con ella, la suma de cualquier rectángulo cuesta 4 lecturas sin importar su tamaño. | `_window_scores` (F5) |
| **Longitud de arco** | La distancia recorrida *a lo largo* de la curva (suma de segmentos pequeños con Pitágoras), no en línea recta. | Clase `Arch` (F5) |
| **Distancia euclidiana** | La distancia recta entre dos puntos: `√((x₂−x₁)²+(y₂−y₁)²)`. | Filtro de diastemas, anchos de piezas (F5, F6) |
| **Porcentaje → pixeles** | Si una pieza ocupa `P %` de la arcada y esta mide `L` px, la pieza mide `P·L/100` px. Así no importa la resolución de la imagen. | Ec. 3.8 (F5) |
| **Matriz de rotación** | Una matriz 2×2 `[[cos α, sin α],[−sin α, cos α]]` que gira puntos α grados. Junto con una traslación forma una transformación afín. | `isolate_arch` (F6) |
