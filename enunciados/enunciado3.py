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
    suma_1 = math.isclose(sum([respuestas[r] for r in respuestas]), 1.0)
    positivas_o_nulas = math.isclose(sum([respuestas[r] < 0 for r in respuestas]), 0.0)
    if not (suma_1 and positivas_o_nulas):
        warnings.warn(
            f"""La respuesta al enunciado {funcion} '{enunciado}' no es una distribución de creencias""",
            RuntimeWarning,
        )

    return suma_1 and positivas_o_nulas


def maxima_incertidumbre(respuestas):
    """
    Si no se proveé una respuesta que sea una distribución de creencias, se construye una dividiendo la creencia en partes iguales.
    """
    n = len(respuestas)
    return {r: 1 / n for r in respuestas}


#######################################
##### Selección Múltiple Semana 3 #####

random.seed(0)


def _3_1(
    enunciado="""Una casa de apuestas paga 3 por Cara y 1.2 por Sello. La moneda tiene 0.5 de probabilidad de que salga Cara o Sello. Si no estamos obligados a apostar todos nuestros recursos cada vez que jugamos, ¿qué proporción conviene apostar a Cara, qué proporción a Sello y qué proporción ahorramos?.""",
):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}

    # Proporción de los recursos que apostamos a Cara
    respuestas["Cara"] = 0.5

    # Proporción de los recursos que apostamos a Sello
    respuestas["Sello"] = 0.0

    # Proporción de los recursos que ahorramos
    respuestas["Ahorro"] = 0.5

    justificacion[
        nombre
    ] = """
    Nunca conviene apostar a sello ya que tiene la misma probabilidad que cara. No conviene apostar todo por si sale sello. Supongo que hay una forma de optimizar la estrategia, pero no se me ocurre.

    """

    # Revisa si es una distribución de creencias (que sume 1)
    valida = es_distribucion_de_creencias(
        respuestas, inspect.currentframe().f_code.co_name, enunciado
    )

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)


def _3_2(
    enunciado="""Cuál es la menor cantidad de preguntas Sí/No que se necesitan para identificar un entero de 0 a 15? """,
):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}
    respuestas["2"] = 0.0
    respuestas["3"] = 0.0
    respuestas["4"] = 1.0
    respuestas["5"] = 0.0
    respuestas["6"] = 0.0
    respuestas["7"] = 0.0
    respuestas["8"] = 0.0
    respuestas["9"] = 0.0
    respuestas["10"] = 0.0
    respuestas["11"] = 0.0
    respuestas["12"] = 0.0
    respuestas["13"] = 0.0
    respuestas["14"] = 0.0
    respuestas["15"] = 0.0

    justificacion[
        nombre
    ] = """
    Cada pregunta binaria divide el espacio a la mitad, y 2^4 = 16 cubre todos los enteros de 0 a 15.
    """

    valida = es_distribucion_de_creencias(
        respuestas, inspect.currentframe().f_code.co_name, enunciado
    )

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)


def _3_3(
    enunciado="""Los modelos se evalúan en función de su capacidad predictiva. El mejor modelo es el que predice los datos con probabilidad 1. ¿Qué tipo de preguntas (o recolección de datos) ofrecen mayor información? ¿Sobre las que no nos generan sorpresa, sobre las que sí nos generan sorpresa o la información de una respuestas (o dato) no depende de la sorpresa?""",
):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}
    respuestas["Sí nos generan sorpresa"] = 1.0
    respuestas["No nos generan sorpresa"] = 0.0
    respuestas["La sorpresa es independiente de la información"] = 0.0

    justificacion[
        nombre
    ] = """
    La información de Shannon mide la sorpresa -log(p): mayor sorpresa -> más información. Por lo tanto, la información es mayor cuando el modelo tiene menor certeza.
    """

    valida = es_distribucion_de_creencias(
        respuestas, inspect.currentframe().f_code.co_name, enunciado
    )

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)


def _3_4(
    enunciado="""El juego submarino es una simplificación del juego 'Batalla Naval'. Hay un tablero de 8x8 y solo una de las celdas contiene al submarino. ¿Obtenemos la misma información si encontramos al submarino en el primer intento que en el n-ésimo intento?
 """,
):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}
    respuestas["Sí, obtenemos la misma información"] = 0.2
    respuestas["No, obtenemos mayor o menor información"] = 0.8

    justificacion[
        nombre
    ] = """
    La información ganada en el n-ésimo intento es -log2(1/m) = log2(m) 'bits', donde m = 65 - n (para n >= 1). La cantidad de información ganada en el evento específico de descubrimiento varía según el contexto de conocimiento previo.
    """

    valida = es_distribucion_de_creencias(
        respuestas, inspect.currentframe().f_code.co_name, enunciado
    )

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)


def _3_5(
    enunciado="""Se dice que un modelo es lineal cuando el modelo solo puede modelar relaciones no lineales entre los datos.""",
):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}
    respuestas["Verdadero"] = 0.0
    respuestas["Falso"] = 1.0

    justificacion[
        nombre
    ] = """
    “Lineal” se refiere a linearidad en parámetros o variables.
    """

    valida = es_distribucion_de_creencias(
        respuestas, inspect.currentframe().f_code.co_name, enunciado
    )

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)


def _3_6(
    enunciado="""Al comparar modelos causales alternativos dado un conjunto de datos, P(M|D), dónde está contenida la información de los datos.""",
):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}
    respuestas["Solo en P(D|M)"] = 0.0
    respuestas["Solo en P(D)"] = 0.0
    respuestas["En ambas en P(D) y P(D|M)"] = 1.0
    respuestas["En todo los elementos de P(M|D)"] = 0.0

    justificacion[
        nombre
    ] = """
    La verosimilitud y la marginal de los datos contienen toda la información aportada por los datos.
    """

    valida = es_distribucion_de_creencias(respuestas, nombre, enunciado)

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)


def _3_7(
    enunciado="""Si la realidad causal subyacente contiene aleatoriedad, ¿contar con el modelo causal probabilístico que se corresponde perfectamente con la realidad causal subyacente permite eliminar la sorpresa completamente (predecir con 1 todos los datos observados)?
 """,
):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}
    respuestas["Sí"] = 0.0
    respuestas["No"] = 1.0

    justificacion[
        nombre
    ] = """
    Con aleatoriedad inherente, nunca es posible predecir todos los datos con probabilidad 1.
    """

    valida = es_distribucion_de_creencias(respuestas, nombre, enunciado)

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)


def _3_8(
    enunciado="""¿Podemos tener esperanza de que alguna vez el avance de la Inteligencia Artificial permita desarrollar modelos que mejoren el desempeño de los modelos causales probabilístico que se corresponde perfectamente con la realidad causal subyacente?
 """,
):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}
    respuestas["Sí"] = 0.0
    respuestas["No"] = 1.0

    justificacion[
        nombre
    ] = """
    Si ya tenemos un modelo causal perfecto ajustado a la realidad, no hay nada que la IA pueda mejorar. No queda espacio para superar ese ideal.
    """

    valida = es_distribucion_de_creencias(respuestas, nombre, enunciado)

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)


def _3_9(
    enunciado="""Tenemos 12 pelotas visualmente iguales. Todas tienen el mismo peso, salvo una que tiene un peso distinto al resto, imperceptible para el ser humano, pero que es suficiente para inclinar una balanza mecánica de dos bandejas. Decidir cómo distribuir las 12 pelotas en el primer uso de la balanza (bandeja izquierda, bandeja derecha, afuera) para garantizar que la balanza sea usada la menor cantidad de veces posibles.""",
):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}
    respuestas["(6, 6, 0)"] = 0.0
    respuestas["(5, 5, 2)"] = 0.0
    respuestas["(4, 4, 4)"] = 1.0
    respuestas["(3, 3, 6)"] = 0.0
    respuestas["(2, 2, 8)"] = 0.0
    respuestas["(1, 1, 10)"] = 0.0

    justificacion[
        nombre
    ] = """
    El peor caso es usar la balanza 3 veces con (4,4,4).  
    """

    valida = es_distribucion_de_creencias(respuestas, nombre, enunciado)

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)


def _3_10(
    enunciado="""Supongamos que sabemos de las 12 pelotas solo la 1, 2, 3 o 4 puede ser la pelota que pesa menos, o que la 5, 6, 7 o 8 la pelota que pesa más. Decidir qué pelotas poner en la balanza izquierda y en derecha para garantizar que la balanza sea usada la menor cantidad de veces posibles.""",
):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}
    respuestas["{1,2,6} vs {3,4,5}"] = 0.0
    respuestas["{1,2,5,6} vs {3,4,7,8}"] = 0.0
    respuestas["{1,6} vs {3,7}"] = 1.0

    justificacion[
        nombre
    ] = """
    Con 2 vs 2 (dejando 4 afuera), un resultado < o > identifica inmediatamente la pelota y si es más liviana o pesada. Si empata, quedan 4 candidatas para la segunda pesada.
    """

    valida = es_distribucion_de_creencias(respuestas, nombre, enunciado)

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)


if __name__ == "__main__":
    print(_3_1())
    print(_3_2())
    print(_3_3())
    print(_3_4())
    print(_3_5())
    print(_3_6())
    print(_3_7())
    print(_3_8())
    print(_3_9())
    print(_3_10())