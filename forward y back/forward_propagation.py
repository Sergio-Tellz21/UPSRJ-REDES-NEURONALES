import numpy as np
import random
from perceptron.perceptron import InputData, Perceptron

def forward_propagation_network(inputs: np.ndarray, perceptrons: int, layers: int) -> float:
    print("corriendo red con los siguientes parámetros:\n- entradas: {inputs}\n- perceptrones por capa: {perceptrons}\n- capas: {layers}\n".format(
        inputs=inputs, perceptrons=perceptrons, layers=layers
    ))

    # Convertir entradas a objetos InputData con peso inicial aleatorio
    current_inputs = [InputData(x) for x in inputs]

    # Construir capas ocultas
    for _ in range(layers):
        neurons = []
        for _ in range(perceptrons):
            b = random.uniform(-1, 1)  # sesgo aleatorio
            neuron = Perceptron(current_inputs, b)
            neurons.append(neuron)
        # Preparar entradas para la siguiente capa
        current_inputs = [InputData(neuron.a) for neuron in neurons]

    # Capa de salida: una sola neurona
    b = random.uniform(-1, 1)
    output_neuron = Perceptron(current_inputs, b)

    return float(output_neuron.a)
