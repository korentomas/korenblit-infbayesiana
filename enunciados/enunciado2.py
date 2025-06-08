import math
import random
import inspect
import warnings

justificacion = {}

# El examen consiste en distribuir creencias entre las hipótesis mutuamente contradictorias en cada una de las afirmaciones o pregunta que se realizan.

def es_distribucion_de_creencias(respuestas, funcion, enunciado):
    """
    Verifica que la respuesta sea una distribución de creencias.
    """
    suma_1 = math.isclose(
        sum([respuestas[r] for r in respuestas]),
        1.0
    )
    positivas_o_nulas = math.isclose(
        sum([respuestas[r]<0 for r in respuestas]),
        0.0
    )
    if not (suma_1 and positivas_o_nulas):
      warnings.warn(f"""La respuesta al enunciado {funcion} '{enunciado}' no es una distribución de creencias""", RuntimeWarning)

    return (suma_1 and positivas_o_nulas)

def maxima_incertidumbre(respuestas):
    """
    Si no se proveé una respuesta que sea una distribución de creencias, se construye una dividiendo la creencia en partes iguales.
    """
    n = len(respuestas)
    return {r: 1/n for r in respuestas}

#######################################
##### Selección Múltiple Semana 2 #####

random.seed(0)

def _2_1(enunciado = """Una casa de apuestas paga 3 por Cara y 1.2 por Sello. La moneda tiene 0.5 de probabilidad de que salga Cara o Sello. Si estamos obligados a apostar todos nuestros recursos cada vez que jugamos, ¿qué proporción conviene apostar a Cara y qué proporción a Sello?. Notar que si apostamos todo a Cara y la moneda resulta que sale Sello, perdemos todos los recursos y no vamos a poder volver a jugar, nos quedamos en cero."""):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}

    # Proporción de los recursos que apostamos a Cara
    respuestas["Cara"] = 0.0

    # Proporción de los recursos que apostamos a Sello
    respuestas["Sello"] = 0.0

    justificacion[nombre] = """
    Agregue su justificación opcional aquí.
    """

    # Revisa si es una distribución de creencias (que sume 1)
    valida = es_distribucion_de_creencias(
        respuestas,
        inspect.currentframe().f_code.co_name,
        enunciado)

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)


def _2_2(enunciado = """El tablero de Galton es un dispositivo que permite hacer una aproximación discreta de la distribución normal mediante la caída de bolas a través de una serie de obstáculos regulares. ¿Qué representa cada uno de los obstáculos?"""):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}
    respuestas["Una distribución Uniforme"] = 0.0
    respuestas["Una distribución Bernoulli"] = 0.0
    respuestas["Una distribución Binomial"] = 0.0
    respuestas["Una distribución Determinista"] = 0.0
    respuestas["Ninguna de las anteriores"] = 0.0

    justificacion[nombre] = """
    Agregue su justificación opcional aquí.
    """

    valida = es_distribucion_de_creencias(
        respuestas,
        inspect.currentframe().f_code.co_name,
        enunciado)

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)

def _2_3(enunciado = """El tablero de Galton es un dispositivo que permite hacer una aproximación discreta de la distribución normal mediante la caída de bolas a través de una serie de obstáculos regulares. ¿Qué representan cada uno de los recipientes en los que se acumulan las bolas al final del recorrido?"""):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}
    respuestas["Las probabilidades individuales de cada trayectoria de las bolas"] = 0.0
    respuestas["El valor de una variable aleatoria Gaussiana"] = 0.0
    respuestas["La varianza de la distribución Gaussiana"] = 0.0
    respuestas["El valor de una variable aleatoria Binomial"] = 0.0
    respuestas["Ninguna de las anteriores"] = 0.0

    justificacion[nombre] = """
    Agregue su justificación opcional aquí.
    """

    valida = es_distribucion_de_creencias(
        respuestas,
        inspect.currentframe().f_code.co_name,
        enunciado)

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)

def _2_4(enunciado = """Tenemos que comprar un producto y debemos elegir un negocio entre tres. Para decidir vamos a usar la información de $n$ consumidores, $i \in \{1, ..., n\}$, reales verificados que luego de hacer la compra indicaron "Me Gustó" o "No me gustó", $m_i \in \{0,1\}$. Vamos a suponer un modelo simple, en el cuál cada negocio $j$ puede ser descrito mediante una variable $p_j \in [0,1]$, que representa la probabilidad que a sus consumidores les guste el producto adquirido, $P(m_i|p)$. Suponer que ya hemos calculado el posterior $P(p|Datos = {m_1, ..., m_n})$ para cada uno de los negocios. ¿Cuál es la forma correcta de decidir en qué negocio comprar?.
 """):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}
    respuestas["Elegir el negocio k para el cual P(p_k|Datos) tenga mayor mediana"] = 0.0
    respuestas["Elegir el negocio k para el cual P(m|p_k,Datos) sea mayor"] = 0.0
    respuestas["Elegir el negocio k comparando visualmente las distribución de creencias P(p_k|Datos)"] = 0.0
    respuestas["Elegir el negocio k para el cual P_k(m|Datos) sea mayor"] = 0.0
    respuestas["Elegir el negocio k para el cual P(p_k|Datos) tenga menor desvío estándar"] = 0.0
    respuestas["Ninguna de las anteriores"] = 0.0

    justificacion[nombre] = """
    Agregue su justificación opcional aquí.
    """

    valida = es_distribucion_de_creencias(
        respuestas,
        inspect.currentframe().f_code.co_name,
        enunciado)

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)

def _2_5(enunciado = """Se dice que un modelo es lineal cuando el modelo solo puede modelar relaciones no lineales entre los datos."""):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}
    respuestas["Verdadero"] = 0.0
    respuestas["Falso"] = 0.0

    justificacion[nombre] = """
    Agregue su justificación opcional aquí.
    """

    valida = es_distribucion_de_creencias(
        respuestas,
        inspect.currentframe().f_code.co_name,
        enunciado)

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)

def _2_6(enunciado = """A pesar de que apliquemos estrictamente las reglas de la probabilidad, cuando dividimos el conjunto de datos de forma aleatoria en dos partes (el primero llamado entrenamiento y el segundo llamado testeo), casi siempre ocurre que la predicción sobre el conjunto de entrenamiento es mejor que la predicción sobre el conjunto de testeo."""):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}
    respuestas["Verdadero"] = 0.0
    respuestas["Falso"] = 0.0

    justificacion[nombre] = """
    Agregue su justificación opcional aquí.
    """

    valida = es_distribucion_de_creencias(
        respuestas,
        nombre,
        enunciado)

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)

def _2_7(enunciado = """A diferencia de lo que suele ocurrir con los modelos de Inteligencia Artificial, cuando una persona comienza a entrenarse en una nueva tarea suele tener un desempeño bajo respecto del desempeño que esa misma persona exhibe una vez que ha finalizado el proceso de entrenamiento y es testeada.
 """):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}
    respuestas["Verdadero"] = 0.0
    respuestas["Falso"] = 0.0

    justificacion[nombre] = """
    Agregue su justificación opcional aquí.
    """

    valida = es_distribucion_de_creencias(
        respuestas,
        nombre,
        enunciado)

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)


def _2_8(enunciado = """Durante el siglo 20 se propuso por primera vez la técnica de validación cruzada para evaluar modelos alternativos."""):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}
    respuestas["La validación cruzada introdujo una corrección necesaria a las reglas de la probabilidad tal como se las conocía desde finales del siglo 18."] = 0.0
    respuestas["La validación cruzada no es necesaria cuando se aplica estrictamente las reglas de la probabilidad propuestas a finales del siglo 18."] = 0.0

    justificacion[nombre] = """
    Agregue su justificación opcional aquí.
    """

    valida = es_distribucion_de_creencias(
        respuestas,
        nombre,
        enunciado)

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)

def _2_9(enunciado = """Es verdadero o falso que, bajo las reglas de la probabilidad, la predicción que hace un modelo respecto de un conjunto de datos, $P(Datos = {d_1, ..., d_n}|Modelo)$, se puede descomponer sin importar el orden de la descomposición como por ejemplo $P(Datos|Modelo) = P(d_3|Modelos)P(d_n|d_3, Modelo)P(d_1|d_n,d_3,Modelo) ... $ donde cada nuevo dato del conjunto se predice dado los datos previos."""):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}
    respuestas["Verdadero"] = 0.0
    respuestas["Falso"] = 0.0

    justificacion[nombre] = """
    Agregue su justificación opcional aquí.
    """

    valida = es_distribucion_de_creencias(
        respuestas,
        nombre,
        enunciado)

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)

def _2_10(enunciado = """Sabemos que es verdadero que la predicción que hace un modelo respecto de un conjunto de datos, $P(Datos = {d_1, ..., d_n}|Modelo)$, se puede descomponer siguiendo estrictamente el orden con el que se fueron recibiendo los datos $P(Datos|Modelo) = P(d_1|Modelos)P(d_2|d_1, Modelo) ...  P(d_n|d_1, ..., d_{n-1}, Modelo)$. ¿Por qué vale esta descomposición?."""):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}
    respuestas["La descomposición es válida porque cada dato es independiente de los demás dado el modelo."] = 0.0
    respuestas["La descomposición es válida porque cada predicción es un posterior y por lo tanto su denominador se cancela con la predicción previa."] = 0.0
    respuestas["La descomposición es válida por la propiedad conmutativa de la multiplicación."] = 0.0
    respuestas["La descomposición es válida pero por ninguna de las razones anteriores."] = 0.0

    justificacion[nombre] = """
    Agregue su justificación opcional aquí.
    """

    valida = es_distribucion_de_creencias(
        respuestas,
        nombre,
        enunciado)

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)

def _2_11(enunciado = """Bajo las reglas de la probabilidad los modelos se los evalúa en función de su desempeño predictivo.  ¿Qué ocurre con la evaluación de un modelo $A$ que predice todos los datos con probabilidad 1 y tan sólo uno con probabilidad 0 respecto de otro modelo $B$ que predice todo los datos con probabilidad 0.5?. Tener en cuenta que el modelo $A$ tiene mayor accuracy, recall, precision y F-score que el modelo $B$."""):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}
    #
    respuestas["Debido a que el modelo $A$ tiene mayor accuracy, recall, precision y F-score que el modelo $B$, podemos decir con certeza que el modelo $A$ tiene mejor desempeño predictivo que el modelo $B$."] = 0.0
    #
    respuestas["A pesar de que el modelo $A$ tiene mayor accuracy, recall, precision y F-score que el modelo $B$, podemos decir con certeza de que el modelo $A$ tiene peor desempeño predictivo que el modelo $B$."] = 0.0

    justificacion[nombre] = """
    Agregue su justificación opcional aquí.
    """

    valida = es_distribucion_de_creencias(
        respuestas,
        nombre,
        enunciado)

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)

def _2_12(enunciado = """Bajo las reglas de la probabilidad, los modelos realizan cada una de las predicciones con la contribución de todas sus hipótesis internas. Por ejemplo, $P(d_{n+1}|Datos_{1:n}, Modelo) = \sum_{Hipótesis \in Modelo} P(d_{n+1}|Hipótesis,Datos_{1:n},Modelo)P(Hipótesis|Datos_{1:n}, Modelo)$. ¿Sería o no sería riesgoso para el modelo seleccionar la hipótesis que optimiza la función objetivo para predecir nuevos datos solo en base a ella sin la contribución del resto de las hipótesis? ¿Por qué?"""):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}
    #
    respuestas["No hay riesgo para el modelo cuando se selecciona la hipótesis que optimizan la función objetivo. Predecir solo con ella ofrece mejores resultados que predecir incluyendo la contribución del resto de las hipótesis que no son óptimas."] = 0.0
    #
    respuestas["Sí hay riesgo para el modelo cuando se selecciona la hipótesis que optimiza la función objetivo. Predecir con la contribución de todas las hipótesis significa sumar las predicciones individuales, lo que ayuda a evitar predicciones cercanas al cero."] = 0.0
    #
    respuestas["Ninguna de las anteriores"] = 0.0

    justificacion[nombre] = """
    Agregue su justificación opcional aquí.
    """

    valida = es_distribucion_de_creencias(
        respuestas,
        nombre,
        enunciado)

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)


def _2_13(enunciado = """Supongamos que hay una realidad causal subyacente y que tenemos dos modelos causales alternativos. El modelo $A$ se corresponde perfectamente con la realidad causal subyacente mientras que el modelo $B$ tiene invertida una de las relaciones causales. Necesariamente el modelo $A$ va a tener mejor o igual desempeño predictivo que el modelo $B$."""):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}

    respuestas["Verdadero"] = 0.0
    respuestas["Falso"] = 0.0

    justificacion[nombre] = """
    Agregue su justificación opcional aquí.
    """

    valida = es_distribucion_de_creencias(
        respuestas,
        nombre,
        enunciado)

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)


if __name__ == '__main__':
    print(_2_1())
    print(_2_2())
    print(_2_3())
    print(_2_4())
    print(_2_5())
    print(_2_6())
    print(_2_7())
    print(_2_8())
    print(_2_9())
    print(_2_10())
    print(_2_11())
    print(_2_12())
    print(_2_13())
