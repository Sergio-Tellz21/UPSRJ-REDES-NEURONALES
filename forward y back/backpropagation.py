import numpy as np
import random
from perceptron.perceptron import InputData, Perceptron

def backpropagation_network(inputs: np.ndarray, perceptrons: int, layers: int) -> float:
    print("corriendo red de retropropagación con los siguientes parámetros:\n- entradas: {inputs}\n- perceptrones por capa: {perceptrons}\n- capas: {layers}\n".format(
        inputs=inputs, perceptrons=perceptrons, layers=layers
    ))

    learning_rate = 0.5
    epochs = 50

    # Para pruebas, valor esperado de salida (puedes cambiar lógica según test)
    expected_output = 1.0 if np.any(inputs) else 0.0

    # Convertir entradas a InputData (solo 2 argumentos)
    current_inputs = [InputData(x) for x in inputs]

    for _ in range(epochs):
        layer_inputs = current_inputs
        layers_neurons = []

        # Forward propagation
        for _ in range(layers):
            neurons = []
            for _ in range(perceptrons):
                b = random.uniform(-1, 1)
                neuron = Perceptron(layer_inputs, b)
                neurons.append(neuron)
            layers_neurons.append(neurons)
            # Preparar entradas para siguiente capa (solo 2 argumentos)
            layer_inputs = [InputData(neuron.a) for neuron in neurons]

        # Capa de salida
        output_neuron = Perceptron(layer_inputs, random.uniform(-1,1))

        # Calcular error y delta de salida
        error = expected_output - output_neuron.a
        delta_output = error * output_neuron.a * (1 - output_neuron.a)

        # Actualizar pesos de la capa de salida
        for inp in output_neuron.inputs:
            inp.w += learning_rate * delta_output * inp.x
        output_neuron.b += learning_rate * delta_output

        # Actualizar pesos de capas ocultas
        for neurons in reversed(layers_neurons):
            for neuron in neurons:
                delta_hidden = sum(delta_output * out_inp.w for out_inp in output_neuron.inputs) * neuron.a * (1 - neuron.a)
                for inp in neuron.inputs:
                    inp.w += learning_rate * delta_hidden * inp.x
                neuron.b += learning_rate * delta_hidden

    return float(output_neuron.a)
