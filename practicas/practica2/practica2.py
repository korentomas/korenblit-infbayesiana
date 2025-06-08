# %% [markdown]
# # Práctica 2: Inferencia Exacta - Ventajas y Límites
# 
# Docente: Gustavo Landfried
# Inferencia Bayesiana Causal 1
# 1er cuatrimestre 2025
# UNSAM
# 
# Alumno: Tomás Pablo Korenblit

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
from scipy.integrate import quad
from statsmodels.api import OLS
from ModeloLineal import moments_posterior, log_evidence
import seaborn as sns

# Configuración global de matplotlib
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.facecolor': 'whitesmoke',
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'axes.titlecolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'legend.frameon': True,
    'legend.facecolor': 'white',
    'legend.edgecolor': 'black',
    'legend.framealpha': 0.8,
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})

# %% [markdown]
# ## 1. ¿En qué negocio conviene invertir?
# 
# En esta sección analizaremos diferentes oportunidades de negocio y evaluaremos sus probabilidades de éxito. Utilizaremos un enfoque bayesiano para cuantificar la incertidumbre y tomar decisiones informadas sobre dónde invertir nuestros recursos.

# %%
np.random.seed(16)
negocios = {
    "A": {"k": 10, "n": 10},
    "B": {"k": 48, "n": 50},
    "C": {"k": 186, "n": 200}
}

def Pp(p):
    """Prior uniforme en (0,1)"""
    return 1

def Pk_np(k, n, p):
    """Verosimilitud binomial"""
    return st.binom.pmf(k, n, p)

# %% [markdown]
# ## 1.1 Interpretación del Galton Board
# 
# El Galton Board nos ofrece una intuición poderosa sobre cómo se acumulan las decisiones binarias. En nuestro contexto:
# - Cada obstáculo representa una decisión binaria: "me gusta" o "no me gusta"
# - Los recipientes en la base representan el número total de "me gusta" posibles
# - La altura del tablero (número de niveles) corresponde al número de personas que toman la decisión
# 
# Esta analogía nos ayuda a visualizar cómo las probabilidades se distribuyen a medida que aumenta el número de observaciones.

# %% [markdown]
# ## 1.2 Visualización de la Distribución Posterior
# 
# Ahora visualizaremos la distribución posterior para cada negocio. Esta distribución nos muestra cómo nuestras creencias sobre la probabilidad de éxito se actualizan después de observar los datos. Cuanto más concentrada sea la distribución, mayor será nuestra certeza sobre la probabilidad de éxito.

# %%
def Pp_kn(p, k, n):
    """Posterior beta"""
    return st.beta.pdf(p, k + 1, n - k + 1)

# Vector de p
p_values = np.linspace(0.6, 1, 100)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for nombre, datos in negocios.items():
    k, n = datos["k"], datos["n"]
    ax.plot(p_values, Pp_kn(p_values, k, n), label=f"{nombre}: k={k}, n={n}", linewidth=2)

ax.set_title("Posterior de $p_i$")
ax.set_xlabel("$p_i$")
ax.set_ylabel("Densidad")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 1.3 Intervalos de Credibilidad
# 
# Los intervalos de credibilidad nos permiten cuantificar nuestra incertidumbre sobre la probabilidad de éxito de cada negocio. Un intervalo del 95% significa que, según nuestro modelo y los datos observados, hay un 95% de probabilidad de que la verdadera probabilidad de éxito se encuentre dentro de ese rango.

# %%
# Calcular intervalos de credibilidad
intervalos = [
    st.beta.interval(0.95, datos["k"] + 1, datos["n"] - datos["k"] + 1)
    for datos in negocios.values()
]

# Etiquetas y posiciones
labels = list(negocios.keys())
x = np.arange(len(labels))

# Graficar
fig, ax = plt.subplots(figsize=(8, 5))
for i, (low, high) in enumerate(intervalos):
    ax.plot([i, i], [low, high], marker="o", color="black", linewidth=2, markersize=8)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Intervalo de credibilidad del 95%")
ax.set_title("Intervalos de credibilidad del 95% para $p_i$")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 1.4 Toma de Decisiones
# 
# Finalmente, calcularemos la probabilidad marginal de éxito para cada negocio. Esta probabilidad nos ayudará a tomar una decisión informada sobre dónde invertir nuestros recursos, considerando tanto los datos observados como nuestra incertidumbre.

# %%
# Probabilidad marginal de m = 1
prob_marginal_m = []

for nombre, datos in negocios.items():
    k, n = datos["k"], datos["n"]
    integral, _ = quad(lambda p: p * Pp_kn(p, k, n), 0, 1)
    prob_marginal_m.append((nombre, integral))

# Mostrar resultados
for nombre, p in prob_marginal_m:
    print(f"Probabilidad marginal de m=1 para el negocio {nombre}: {p:.3f}")

# %% [markdown]
# ## 2. La Puntería de las Arqueras Mexicanas
# 
# En esta sección analizaremos un problema clásico de inferencia bayesiana: determinar la posición del arco basándonos en las posiciones de las flechas. Este ejemplo nos permitirá ver cómo la inferencia bayesiana nos ayuda a actualizar nuestras creencias de manera secuencial.
# 
# Variables del modelo:
# - $x$ = posición del arco (incógnita)
# - $y_i$ = posición de la flecha (observable)
# - $\beta$ = desviación estándar de la posición de la flecha
# - $\sigma$ = desviación estándar de la posición del arco
# - $\mu$ = posición promedio del arco
# 
# Modelo probabilístico:
# - Posición de la flecha: $P(y_i|x) = N(y_i|x,\beta^2)$
# - Prior sobre la posición del arco: $P(x) = N(x|\mu,\sigma^2)$

# %% [markdown]
# ## 2.1 Predicción de la Primera Flecha
# 
# Comenzaremos analizando la distribución predictiva para la primera flecha. Esta distribución nos muestra todas las posiciones posibles de la flecha, considerando nuestra incertidumbre sobre la posición del arco.

# %%
# Prior sobre x: N(0, 1.5)
def Px(x):
    """Prior sobre la posición del arco"""
    return st.norm.pdf(x, 0, 1.5)

# Likelihood: y ~ N(x, 1)
def Py_x(y, x):
    """Verosimilitud de la posición de la flecha dado el arco"""
    return st.norm.pdf(y, x, 1)

def aproximar_conjunta(rango):
    """Aproximación de la distribución conjunta P(x, y)"""
    Z = np.zeros((len(rango), len(rango)))
    for i, x in enumerate(rango):
        for j, y in enumerate(rango):
            Z[j, i] = Py_x(y, x) * Px(x)
    return Z

# Rango de evaluación
valores = np.linspace(-5, 5, 200)
Z = aproximar_conjunta(valores)

# Gráfico
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Distribución conjunta aproximada
contour = ax[0].contourf(valores, valores, Z, levels=7, cmap="Greys")
ax[0].contour(valores, valores, Z, levels=7, colors="black", linewidths=0.5)
plt.colorbar(contour, ax=ax[0], label="Densidad")

ax[0].set_title("Distribución conjunta aproximada de $x$ e $y$")
ax[0].set_xlabel("$x$ (posición del arco)")
ax[0].set_ylabel("$y$ (posición de la flecha)")

# Línea en y = y_val
y_val = 1.33
i_y = np.abs(valores - y_val).argmin()
ax[0].axhline(y=y_val, color="black", linestyle="dotted")

# Distribución marginal aproximada de y
Z_y = Z[i_y, :]
ax[1].plot(valores, Z_y, color="black", linewidth=2)
ax[1].fill_between(valores, Z_y, color="grey", alpha=0.5)

ax[1].set_title(f"Distribución marginal aproximada de $y$ con $y={y_val}$")
ax[1].set_xlabel("$x$")
ax[1].axvline(x=y_val, color="black", linestyle="dotted")

# Área bajo la curva: P(y) ≈ ∫ p(y|x)p(x) dx
integral, _ = quad(lambda x: Py_x(y_val, x) * Px(x), -5, 5)
ax[1].text(
    y_val,
    max(Z_y) / 3,
    f"P(y) ≈ {integral:.3f}",
    fontsize=12,
    color="darkred",
    ha="center"
)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2.2 Actualización de Creencias
# 
# A medida que observamos más flechas, actualizamos nuestra creencia sobre la posición del arco. Veremos cómo la distribución posterior se vuelve más precisa y concentrada con cada nueva observación.

# %%
def Px_y(y, mu=0, sigma=3, beta=1):
    """Devuelve los parámetros de la distribución posterior de x dado y"""
    mu_post = (y * sigma**2 + mu * beta**2) / (sigma**2 + beta**2)
    sigma_post = np.sqrt((sigma**2 * beta**2) / (sigma**2 + beta**2))
    return mu_post, sigma_post

# Rango para graficar
x_range = np.linspace(-2, 6, 200)

# Gráficos
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Prior inicial sobre x
axs[0].plot(x_range, st.norm.pdf(x_range, loc=0, scale=3),
            label="Prior sobre $x$", color="black", linewidth=2)

# Actualización bayesiana secuencial
for i, yi in enumerate(posiciones_flecha):
    # Inicializar con prior si es el primer tiro
    if i == 0:
        mu_post, sigma_post = Px_y(yi)
    else:
        mu_post, sigma_post = Px_y(yi, mu_post, sigma_post)

    # Graficar algunas posteriores seleccionadas
    if i % 12 == 0:
        axs[0].plot(
            x_range,
            st.norm.pdf(x_range, mu_post, sigma_post),
            label=f"i={i}",
            alpha=0.75,
            linewidth=2
        )

    # Media y desviación estándar en función de i
    axs[1].plot(i, mu_post, "o", color="black", alpha=0.5, markersize=6)
    axs[1].plot(
        [i, i],
        [mu_post - sigma_post, mu_post + sigma_post],
        color="dimgray", alpha=0.3,
        linewidth=2
    )

# Línea en la media real de los datos generados
axs[1].axhline(y=2, color="black", linestyle="--", linewidth=2)

# Estética
axs[0].legend()
axs[0].set_xlabel("Posición del arco [$x$]")
axs[0].set_ylabel("Densidad")
axs[0].set_title("Distribución posterior de $x$ dado ${y_0, ..., y_i}$")

axs[1].set_ylabel("Posición del arco [$x$]")
axs[1].set_xlabel("Tiro de la flecha $i$")
axs[1].set_title("Media y desvío estándar de $p(x|\{y_0, ..., y_i\})$")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2.3 Predicción de la Próxima Flecha
# 
# Finalmente, utilizaremos nuestro conocimiento actualizado para predecir la posición de la próxima flecha. Esta predicción incorpora toda la incertidumbre que tenemos sobre la posición del arco.

# %%
# Posterior final sobre x (después de todos los datos observados)
mu_final, sigma_final = mu_post, sigma_post  # últimos valores tras loop anterior

def posterior_x(x):
    """Densidad posterior final sobre x"""
    return st.norm.pdf(x, mu_final, sigma_final)

def posterior_predictiva_y(y):
    """Distribución predictiva de y dado todos los datos"""
    return quad(lambda x: Py_x(y, x) * posterior_x(x), -6, 6)[0]

# Evaluar para un rango de y
y_range = np.linspace(-6, 6, 200)
predictiva = [posterior_predictiva_y(y) for y in y_range]

# Graficar
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y_range, predictiva, label="Posterior predictiva", color="black", linewidth=2)
ax.set_xlabel("Posición de la flecha [$y$]")
ax.set_ylabel("Densidad")
ax.set_title("Distribución posterior predictiva de $y_{i+1}$")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Modelos Polinomiales: Encontrando el Balance
# 
# En esta sección exploraremos cómo la inferencia bayesiana nos ayuda a encontrar el balance adecuado entre la complejidad del modelo y su capacidad predictiva. Analizaremos modelos polinomiales de diferentes grados y veremos cómo la evidencia del modelo nos guía en la selección del modelo más apropiado.

# %% [markdown]
# ## 3.1 Generación de Datos
# 
# Comenzaremos generando datos que siguen una función sinusoidal con ruido gaussiano. Estos datos nos servirán para evaluar cómo los diferentes modelos polinomiales se ajustan a patrones no lineales.

# %%
n_obs = 30        # cantidad de observaciones
beta = 0.2        # desviación estándar del ruido
np.random.seed(222)

def f(x):
    """Función sinusoidal subyacente"""
    return np.sin(2 * np.pi * x)

def simular_datos(f, n, beta, xmin=-0.5, xmax=0.5):
    """Genera datos con ruido gaussiano alrededor de f(x)"""
    x = np.random.uniform(xmin, xmax, n)
    y = np.random.normal(loc=f(x), scale=beta)
    return x, y

def graficar_datos(x, y, f, xmin=-0.5, xmax=0.5):
    """Visualiza los datos simulados y la función subyacente"""
    x_eval = np.linspace(xmin, xmax, 200)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_eval, f(x_eval), label="Función subyacente", linestyle="dotted", color="black", linewidth=2)
    ax.plot(x, y, "o", label="Datos simulados", color="black", alpha=0.5, markersize=8)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Simulación de datos con función sinusoidal")
    ax.legend()
    plt.tight_layout()
    plt.show()

x, y = simular_datos(f, n_obs, beta)
graficar_datos(x, y, f)

# %% [markdown]
# ## 3.2 Análisis de Verosimilitud
# 
# Analizaremos la log-verosimilitud de los diferentes modelos polinomiales. Este análisis nos mostrará cómo la verosimilitud aumenta con la complejidad del modelo, pero también nos alertará sobre el riesgo de sobreajuste.

# %%
def transformacion_polinomica(x, grado):
    """Transforma x en una matriz de diseño polinomial"""
    return np.array([x**i for i in range(grado + 1)]).T

fig, ax = plt.subplots(figsize=(10, 6))

for grado in range(10):
    X_poly = transformacion_polinomica(x, grado)
    modelo = OLS(y, X_poly).fit()
    ax.bar(grado, modelo.llf, alpha=0.5, label=f"Grado {grado}")

ax.set_xlabel("Grado del polinomio")
ax.set_ylabel("Log-verosimilitud")
ax.set_title("Log-verosimilitud de los modelos polinómicos")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3.3 Visualización de Ajustes
# 
# Visualizaremos cómo los diferentes modelos polinomiales se ajustan a los datos. Esto nos permitirá ver de manera intuitiva el trade-off entre ajuste y generalización.

# %%
x_min, x_max = x.min(), x.max()
padding = (x_max - x_min) * 0.1
x_range = np.linspace(x_min - padding, x_max + padding, 300)

fig, ax = plt.subplots(1, 2, figsize=(15, 6))

ax[0].plot(x, y, "o", color="black", alpha=0.5, markersize=8)
ax[0].set_ylim(-1.5, 1.5)

ax[1].plot(x_range, f(x_range), color="black", label="Función subyacente", linestyle="dotted", linewidth=2)
ax[1].plot(x, y, "o", color="black", alpha=0.5, markersize=8)
ax[1].set_ylim(-1.5, 1.5)

for grado in range(10):
    modelo = OLS(y, transformacion_polinomica(x, grado)).fit()
    y_hat = modelo.predict(transformacion_polinomica(x_range, grado))
    
    ax[0].plot(x_range, y_hat, label=f"Grado {grado}", alpha=0.5, linewidth=2)

    if grado == 9:
        ax[1].plot(x_range, y_hat, label="Grado 9", alpha=0.7, linewidth=2)

ax[0].set_xlabel("$x$")
ax[0].set_ylabel("$y$")
ax[0].set_title("Ajuste de los modelos polinómicos")
ax[0].legend(ncols=2, fontsize=8)

ax[1].set_xlabel("$x$")
ax[1].set_ylabel("$y$")
ax[1].set_title("Ajuste del modelo polinómico de máxima verosimilitud")
ax[1].legend(ncols=1, fontsize=9)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3.4 Evaluación Predictiva
# 
# Evaluaremos el desempeño predictivo de los modelos en datos nuevos. Esta evaluación nos mostrará que un mejor ajuste a los datos de entrenamiento no necesariamente implica mejor capacidad predictiva.

# %%
k = 100
x_out_of_sample = st.uniform.rvs(-0.5, 1, k)
y_out_of_sample = st.norm.rvs(loc=f(x_out_of_sample), scale=beta, size=k)

fig, ax = plt.subplots(figsize=(10, 6))

for grado in range(10):
    modelo = OLS(y, transformacion_polinomica(x, grado)).fit()
    y_hat = modelo.predict(transformacion_polinomica(x_out_of_sample, grado))

    likelihood_y_hat = np.prod(st.norm.pdf(y_out_of_sample, loc=y_hat, scale=beta))
    ax.bar(grado, likelihood_y_hat, label=f"grado {grado}", alpha=0.5)

ax.set_xlabel("Grado del polinomio")
ax.set_ylabel("Verosimilitud de las predicciones")
ax.set_title("$P(new\_data|M_D$)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3.5 Regularización Bayesiana
# 
# Implementaremos regularización bayesiana para controlar la complejidad del modelo. Veremos cómo los priors adecuados nos ayudan a encontrar modelos que generalizan mejor.

# %%
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

ax[0].plot(x, y, "o", color="black", alpha=0.5, markersize=8)
ax[0].set_ylim(-1.5, 1.5)

for grado in range(10):
    means, cov = moments_posterior(1 / (25)**2, 1, y, transformacion_polinomica(x, grado))

    y_curve = transformacion_polinomica(x_range, grado) @ means
    ax[0].plot(x_range, y_curve, label=f"Grado {grado}", alpha=0.6, linewidth=2)

    y_hat = transformacion_polinomica(x_out_of_sample, grado) @ means
    log_likelihood = np.sum(st.norm.logpdf(y_out_of_sample, loc=y_hat, scale=beta))
    ax[1].bar(grado, log_likelihood, alpha=0.5)

ax[0].set_xlabel("$x$")
ax[0].set_ylabel("$y$")
ax[0].set_title("Ajuste de modelos bayesianos penalizados")
ax[0].legend(ncols=2, fontsize=8)

ax[1].set_xlabel("Grado del polinomio")
ax[1].set_ylabel("Log-verosimilitud en datos nuevos")
ax[1].set_title("$\log P(\mathrm{new\\_data} \mid M_D)$")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3.6 Selección de Modelos
# 
# Utilizaremos la evidencia del modelo para seleccionar el grado polinomial más apropiado. Este enfoque nos permitirá encontrar el balance óptimo entre ajuste y complejidad.

# %%
np.random.seed(22)

fig, ax = plt.subplots(figsize=(10, 6))

model_log_evidence = [
    log_evidence(y, transformacion_polinomica(x, grado), 1e-6, 10)
    for grado in range(10)
]
model_prior = [1 / 10] * 10
model_posterior = model_prior * np.exp(model_log_evidence)
model_posterior /= np.sum(model_posterior)

ax.bar(range(10), model_posterior, color="navy", label="Posterior", alpha=0.5)
ax.set_title("Distribución posterior de los modelos")
ax.set_xlabel("Grado del polinomio")
ax.set_ylabel("Probabilidad")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3.7 Interpretación de Resultados
# 
# Analizaremos cómo los diferentes modelos explican los datos y por qué algunos modelos son preferidos sobre otros. Esta interpretación nos ayudará a entender mejor el proceso de selección de modelos.

# %%
np.random.seed(22)

x_test = -0.23

fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# plot data
ax[0].plot(x, y, "o", color="black", alpha=0.5, markersize=8)
ax[0].set_ylim(-1.5, 1.5)

_legend = ["Rígido", "Simple", "Complejo"]

for grade in range(10):
    means, cov = moments_posterior(1e-6, 10, y, transformacion_polinomica(x, grade))

    # plot curve
    y_curve = transformacion_polinomica(x_range, grade) @ means
    ax[0].plot(x_range, y_curve, label=f"grado {grade}", alpha=0.5, linewidth=2)

    # Plot distribution of the posterior at x = x_test
    if grade in [0, 3, 9]:
        y_test_mean = transformacion_polinomica(x_test, grade) @ means
        y_test_var = (
            transformacion_polinomica(x_test, grade)
            @ cov
            @ transformacion_polinomica(x_test, grade).T
        )
        y_test_sd = np.sqrt(y_test_var)

        y_test_range = np.linspace(-2, 0.5, 500)

        y_test_dist = st.norm.pdf(y_test_range, loc=y_test_mean, scale=y_test_sd)

        ax[1].plot(
            y_test_range,
            y_test_dist,
            label=f"{_legend.pop(0)} (grado {grade})",
            alpha=0.5,
            linewidth=2
        )

ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].set_title("Media del posterior con prior no informativo")
ax[0].legend(ncols=2)

ax[1].set_xlabel(f"Y|X={x_test}")
ax[1].set_ylabel("$P(DATOS|M_D$")
ax[1].set_title(f"$P(Y|X={x_test},M_D$)")
ax[1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Efecto Causal del Sexo Biológico sobre la Altura
# 
# En esta sección aplicaremos la inferencia bayesiana para analizar el efecto causal del sexo biológico sobre la altura. Este ejemplo nos permitirá ver cómo la inferencia bayesiana nos ayuda a distinguir entre diferentes hipótesis causales.

# %% [markdown]
# ## 4.1 Exploración de Datos
# 
# Comenzaremos explorando los datos de alturas para entender mejor las relaciones entre las variables y formular hipótesis iniciales.

# %%
data = pd.read_csv(r"../data/alturas.csv")
data.head()
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=data, x="altura_madre", y="altura", hue="sexo", ax=ax)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4.2 Modelos Causales
# 
# Definiremos tres modelos causales alternativos para explicar la relación entre el sexo biológico y la altura:
# 
# 1. Modelo base: considera solo la altura de la madre
# 2. Modelo biológico: incorpora el efecto del sexo biológico
# 3. Modelo de grupos aleatorios: asigna efectos aleatorios a grupos de individuos
# 
# Estos modelos nos permitirán evaluar diferentes hipótesis sobre la causalidad.

# %%
np.random.seed(22)

y = data["altura"]

# Matriz de diseño modelo base
X0 = np.array([np.ones(len(y)), data["altura_madre"]]).T

# Matriz de diseño modelo biológico
X1 = np.array(
    [np.ones(len(y)), data["altura_madre"], (data["sexo"] == "M").astype(int)]
).T

# Matriz de diseño modelo de grupos al azar
N = len(y)
grupos = np.zeros((N, N // 2))
indices = np.random.permutation(N)

for i, idx in enumerate(indices):
    grupos[idx, i // 2] = 1

X2 = np.hstack([data["altura_madre"].values.reshape(-1, 1), grupos])

# %%
fig, ax = plt.subplots(figsize=(10, 6))

_legend = ["Modelo base", "Modelo biológico", "Modelo de grupos al azar"]
ax.set_xlabel("Modelo")
ax.set_xticks(range(len(_legend)))
ax.set_xticklabels(_legend)
ax.set_ylabel("log likelihood")
ax.set_title("log likelihood de los modelos causales")

log_evidence_mj = []
for i, X in enumerate([X0, X1, X2]):
    # alpha "chico" -> prior poco informativo
    # beta "grande" -> ruido en el observable bajo
    log_evidence_i = log_evidence(y, X, 1e-6, 25)
    log_evidence_mj.append(log_evidence_i)
    ax.bar(i, log_evidence_i, label=_legend[i], alpha=0.8)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4.4 Análisis de Predicciones
# 
# Analizaremos las predicciones de cada modelo utilizando la media geométrica. Esta métrica nos ayudará a comparar el desempeño predictivo de los diferentes modelos.

# %%
# GeoMean_M  = ( P(D∣M) ) ** 1/N = exp( log_evidence / N )
for i, ll in enumerate(log_evidence_mj):
    geomean = np.exp(ll / len(y))
    print(f"{_legend[i]} - Media Geométrica: {geomean:.2e}")

# %% [markdown]
# ## 4.5 Conclusiones
# 
# Finalmente, calcularemos las probabilidades posteriores de cada modelo y extraeremos conclusiones sobre el efecto causal del sexo biológico sobre la altura.

# %%
log_prior_m = np.log(np.array([1 / 3] * 3))
log_posterior_m = log_prior_m + np.array(log_evidence_mj)
log_posterior_m -= np.logaddexp.reduce(log_posterior_m)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel("Modelo")
ax.set_xticks(range(len(_legend)))
ax.set_xticklabels(_legend)
ax.set_ylabel("posterior")
ax.set_title("probabilidad posterior de los modelos causales")

for i, lp in enumerate(log_posterior_m):
    ax.bar(_legend[i], np.exp(lp), label=_legend[i], alpha=0.8)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Modelos Causales Alternativos
# 
# En esta sección exploraremos cómo la inferencia bayesiana nos ayuda a distinguir entre diferentes estructuras causales. Analizaremos dos modelos causales alternativos y veremos cómo los datos nos ayudan a determinar cuál es más plausible.

# %% [markdown]
# ## 5.1 Generación de Datos
# 
# Comenzaremos generando datos según el modelo AcausaB. Este modelo nos servirá como base para evaluar diferentes hipótesis causales.

# %%
def gen_acausab(n=1):
    """Genera datos según el modelo AcausaB"""
    _a = st.bernoulli.rvs(0.5, size=n)
    _b = st.bernoulli.rvs(0.95 * _a + 0.05 * (1 - _a))
    return list(zip(_a, _b))

np.random.seed(22)
data = gen_acausab(16)

results = [(i, j) for i in [0, 1] for j in [0, 1]]
_ticks = [str(x) for x in results]

fig, ax = plt.subplots(figsize=(10, 6))

for i, r in enumerate(results):
    count = data.count(r)
    ax.bar(
        _ticks[i],
        count,
        label=f"A={r[0]}, B={r[1]}",
        alpha=0.5,
    )

ax.legend()
ax.set_ylabel("N")
ax.set_title("Histograma de los datos generados")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5.2 Evaluación de Modelos
# 
# Evaluaremos el desempeño predictivo de los modelos causales alternativos. Esta evaluación nos mostrará cómo la inferencia bayesiana nos ayuda a distinguir entre diferentes estructuras causales.

# %%
def likelihood_i(ai, bi):
    """Verosimilitud para un par de observaciones"""
    return 0.95 if ai == bi else 0.05

def likelihood_model_1(data):
    """Verosimilitud para el modelo 1"""
    return np.prod([likelihood_i(ai, bi) for ai, bi in data])

def likelihood_model_2(data):
    """Verosimilitud para el modelo 2"""
    return np.prod([likelihood_i(ai, bi) for ai, bi in data])

likelihood = np.array([likelihood_model_1(data), likelihood_model_2(data)])

print(f"likelihood modelo 1: {likelihood_model_1(data):.3e}")
print(f"likelihood modelo 2: {likelihood_model_2(data):.3e}")

# %% [markdown]
# ## 5.3 Actualización de Creencias
# 
# Actualizaremos nuestras creencias sobre los modelos después de observar los datos. Este proceso nos mostrará cómo la inferencia bayesiana nos permite aprender de los datos de manera sistemática.

# %%
# Prior uniforme sobre ambos modelos
prior = np.array([0.5, 0.5])
posterior = prior * likelihood
posterior /= np.sum(posterior)

print(f"posterior modelo 1: {posterior[0]:.3f}")
print(f"posterior modelo 2: {posterior[1]:.3f}")

# %% [markdown]
# ## 5.4 Conclusiones
# 
# En este caso particular, veremos que ambos modelos tienen la misma distribución conjunta, lo que significa que observar más datos no nos ayuda a distinguir entre ellos. Esta es una limitación importante de la inferencia basada solo en datos observacionales, y nos muestra la importancia de considerar información adicional o realizar experimentos controlados para establecer causalidad.


# %%
