import numpy as np
import random
# Adjust the import path to include the parent directory for py_utils
import sys
import os
from logging import DEBUG, INFO, WARNING, ERROR
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from py_utils.logger.logger import set_logging, clog

# Llamada a la funcion para configurar el logging
set_logging(log_file='single_perceptron.log')

#############################################################################################################################
# Perceptrón de una sola neurona                                                                                            #
# Una neurona toma n entradas (x1, x2, ... xn), las multiplica por sus respectivos pesos (w1, w2, ...wn),                   #
# les suma un sesgo (b), y produce una salida (a) mediante una combinación lineal:                                          #
#                                                                                                                           #
# z = x1 * w1 + x2 * w2 + ... xn * wn + b                                                                                   #
# a = f(z)  # En este ejemplo, aplicamos la sigmoide como función de activación                                             #
#                                                                                                                           #
# Diagrama conceptual:                                                                                                      #
#   x1 ─┐                                                                                                                   #
#       │                                                                                                                   #
#   x2 ─┼─► [ Neurona ] ──► a                                                                                               #
#       │                                                                                                                   #
#      ...      ↑                                                                                                           #
#       │   (w1, w2, ... wn, b)                                                                                             #
#   xn ─┘                                                                                                                   #
#                                                                                                                           #
#                                                                                                                           #
# Este modelo es la base de redes más complejas. Ideal para introducir conceptos como pesos, sesgo y salida lineal.         #
#                                                                                                                           #
# NOTE: https://docs.python.org/3/tutorial/classes.html                                                                     #
#                                                                                                                           #
#############################################################################################################################

# Paso 1: Abstracción de entradas de una neurona.
#
# TODO: Define una clase "InputData" que contenga todos los elementos del diagrama conceptual referentes a la entrada de un perceptrón.
#
# NOTE: * Considera todos los procesos que ocurren dentro de un perceptrón:
#           - Inicialización aleatoria del peso
#           - Actualización de los pesos para propagación
#
#       * Recuerda que la entrada de un perceptrón contiene:
#           - un valor "x"
#           - un peso "w"
#
class InputData:
    def __init__(self, x: float):
        # Parametro de entrada de usuario
        self.x = x
        # Inicializacion aleatoria
        self.w = self.init_weight()
    
    def init_weight(self):
        return random.random()
    
    def update_weight(self, w: float):
        self.w = w
    
clog(InputData)

# Paso 2: Abstracción de una neurona.
#
# TODO: Define una clase "Perceptron" que contenga todos los elementos del diagrama conceptual referentes a un perceptrón.
#
# NOTE: * Considera todos los procesos que ocurren dentro de un perceptrón:
#           - La suma ponderada de las entradas
#           - La aplicación de una función de activación
#
#       * Recuerda que un perceptrón contiene:
#           - una o varias entradas de datos con un peso respectivo.
#           - un sesgo "b"
#           - una suma ponderada de las entradas "z"
#           - una salida "a" definida por su función de activación
#
class Perceptron:
    def __init__(self, inputs: list[InputData], b: float):
        self.inputs = inputs
        self.b = b
        self.z = self.forward()
        self.a = self.activation()
        
    def forward(self):
        z = 0
        for input in self.inputs:
            z = z + (input.x * input.w)
        return z + self.b
    
    def activation(self):
        return 1.0 / (1.0 + np.exp(-self.z))

clog(Perceptron)