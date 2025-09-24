from exercises.redes_neuronales_artificiales.single_perceptron import InputData, Perceptron
import random
import random
# Adjust the import path to include the parent directory for py_utils
import sys
import os
from logging import DEBUG, INFO, WARNING, ERROR
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from py_utils.logger import set_logging, plog

# Llamada a la funcion para configurar el logging
set_logging(log_file='redes_neuronales_artificiales.log')

i1 = InputData(x=0.1)
i2 = InputData(x=0.2)
i3 = InputData(x=0.3)
 
b1 = random.random()

p1 = Perceptron(inputs=[i1, i2], b=b1)
try:
	inputs_x = [inp.x for inp in p1.inputs]
except Exception:
	inputs_x = str(p1.inputs)  # Retroceder en el caso que sean iguales

plog(f"perceptron 1:\n   - inputs x: {inputs_x}\n   - bias: {p1.b}\n   - forward (z): {getattr(p1, 'z', None)}\n   - output (a): {getattr(p1, 'a', None)}\n")

# Ejercicio 1: Red de una neurona.
#
# TODO: Crea una red neuronal de un solo perceptrón y una sola entrada con los objetos previamente creados.
#       Asigna el valor de la salida a "single_layer_perceptron"
#
# NOTE: Toma como referencia el diagrama conceptual de la red.
#    
#   x ──► [ Neurona ] ──► a
#              ↑
#            (w, b)
#
# Crea un perceptrón con una sola entrada (usar uno de los InputData ya creados)
b_single = random.random()
p_single = Perceptron(inputs=[i1], b=b_single)
# Si la clase no calculó la salida aún, fuerza el forward
if getattr(p_single, 'a', None) is None:
    p_single.forward()
single_layer_perceptron = p_single.a

plog(f"Salida de la red de una sola capa: {single_layer_perceptron}", level=ERROR if single_layer_perceptron is None else DEBUG, eol=True)

# Ejercicio 2: Red de dos neuronas y una entrada.
#
# TODO: Crea una red neuronal de dos perceptrones y una sola entrada con los objetos previamente creados.
#       Asigna el valor de la salida a "two_layer_network"
#
# NOTE: Toma como referencia el diagrama conceptual de la red.
#
#   x1 ──► [ Neurona 1 ] ──► a1 ──► [ Neurona 2 ] ──► a2
#              ↑                          ↑
#            (w1, b1)                   (w2, b2)
#
b_h = random.random()
hidden = Perceptron(inputs=[i1], b=b_h)
if getattr(hidden, 'a', None) is None:
    hidden.forward()
# pasar la salida de la neurona oculta como InputData para la siguiente capa
hidden_output_as_input = InputData(x=hidden.a)

b_out = random.random()
output_neuron = Perceptron(inputs=[hidden_output_as_input], b=b_out)
if getattr(output_neuron, 'a', None) is None:
    output_neuron.forward()

two_layer_network = output_neuron.a

plog(f"Salida de la red de dos capas: {two_layer_network}",
     level=ERROR if two_layer_network is None else DEBUG, eol=True)

# Ejercicio 3: Red de dos neuronas y una entrada.
#
# TODO: Crea una red neuronal de dos perceptrones y dos entradas con los objetos previamente creados.
#       Asigna el valor de la salida a "small_network"
#
# NOTE: Toma como referencia el diagrama conceptual de la red.
#
# Entrada         Capa Oculta 1        Salida
#   x1 ─────────► [ Neurona 1 ]  
#       ↘      ↗                ↘
#         ↘  ↗                    ↘
#           x                        ► [ a2 ] 
#         ↗  ↘                    ↗    
#       ↗      ↘                ↗         
#   x2 ─────────► [ Neurona 2 ]
#
# Crear capa oculta con dos perceptrones que tomen las mismas dos entradas
h1 = Perceptron(inputs=[i1, i2], b=random.random())
h2 = Perceptron(inputs=[i1, i2], b=random.random())
for p in (h1, h2):
    if getattr(p, 'a', None) is None:
        p.forward()

# Convertir salidas ocultas en InputData
hd1 = InputData(x=h1.a)
hd2 = InputData(x=h2.a)

# Neurona de salida que lee las dos salidas ocultas
p_out = Perceptron(inputs=[hd1, hd2], b=random.random())
if getattr(p_out, 'a', None) is None:
    p_out.forward()

small_network = p_out.a

plog(f"Salida de la red de pequeña: {small_network}", level=ERROR if small_network is None else DEBUG,eol=True)