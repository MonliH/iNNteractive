# Big list of imports
import tkinter as tk
from tkinter import DISABLED, NORMAL
from tkinter import ttk
import numpy as np
from numpy import sin, cos, tan
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import keras
import time

print("Program By: Jonathan Li\n")


def write_log(text):
    print(text)
    filename = "debug.log"
    file_obj = open(filename, "a")
    file_obj.write(text)


class Layer(ttk.Frame):
    def __init__(self, master, neurons, activation, **options):
        ttk.Frame.__init__(self, master, **options)
        self.neurons = neurons
        self.activation = activation
        lbl = ttk.Label(self, text="Number of neurons: {} | Activation function: {}".format(neurons, activation))
        btn = ttk.Button(self, text='Remove Layer', command=lambda: remove_layer(self))
        lbl.grid(row=0, column=0, padx=30, pady=2)
        btn.grid(row=0, column=1, padx=30, pady=2)

    def get_properties(self):
        return {"activation": self.activation, "neurons": self.neurons}


class AddLayer(ttk.Frame):
    def __init__(self, master, **options):
        ttk.Frame.__init__(self, master, **options)
        neurons = ttk.Entry(self, )
        label8 = ttk.Label(self, text="Number of Neurons: ")
        label9 = ttk.Label(self, text="Activation Function: ")
        activation = ttk.Combobox(self, values=["softmax", "elu", "selu", "softplus", "softsign", "relu", "tanh",
                                                "sigmoid", "hard_sigmoid", "exponential", "linear"])
        add_button = ttk.Button(self, text="Add Layer!", command=lambda: add_layer(neurons.get(), activation.get(), activation, neurons))
        remove_all = ttk.Button(self, text="Remove All Layers", command=clear_layers)
        label8.grid(row=0, column=0, padx=5, pady=10)
        neurons.grid(row=0, column=1, padx=5, pady=10)
        label9.grid(row=0, column=2, padx=5, pady=10)
        activation.grid(row=0, column=3, padx=5, pady=10)
        add_button.grid(row=0, column=4, padx=5, pady=10)
        remove_all.grid(row=0, column=5, padx=5, pady=10)


def add_layer(neurons, activation, obj1, obj2):
    nn_layers.append({"neurons": neurons, "activation": activation})
    refresh_layer()
    layers_list[-1].grid()
    obj2.delete(0, 'end')


def clear_layers():
    for layer_nn_loop in layers_list:
        layer_nn_loop.destroy()
    layers_list.clear()
    nn_layers.clear()
    refresh_layer()
    show_layer()


def remove_layer(obj):
    try:
        for i in range(len(layers_list)):
            if layers_list[i] is obj:
                layers_list[i].destroy()
                del nn_layers[i]
                del layers_list[i]
        for layer_for_loop in layers_list:
            layer_for_loop.destroy()
        layers_list.clear()
        refresh_layer()
        show_layer()

    except IndexError:
        print("an error happened")


def set_optimizer_loss(type_of_value, number):
    if not type_of_value == "":
        values_nn[number] = type_of_value


def nn_settings():
    global menu
    global training_method
    global loss_type
    global learning_rate_nn
    global values_nn
    m = tk.Toplevel()
    m.config(background="gray91")
    m.title("Change Neural Network Settings")

    menu = ttk.Frame(m)
    menu.grid(row=0, pady=10)

    options_frame = ttk.Frame(m)
    training_method = ttk.Combobox(options_frame, values=["SGD", "RMSprop", "Adagrad", "Adadelta", "Adam", "Adamax", "Nadam"],
                                   state="readonly")
    loss_type = ttk.Combobox(options_frame, values=["mean_squared_error", "mean_absolute_error",
                                                    "mean_absolute_percentage_error", "mean_squared_logarithmic_error",
                                                    "logcosh"], state="readonly", width=25)

    command_change_loss_type = lambda x: set_optimizer_loss(loss_type.get(), 1)
    command_change_training_method = lambda x: set_optimizer_loss(training_method.get(), 0)

    loss_type.bind("<<ComboboxSelected>>", command_change_loss_type)
    training_method.bind("<<ComboboxSelected>>", command_change_training_method)

    #learning_rate_nn = ttk.Entry(options_frame)
    print(values_nn)
    training_method.set(values_nn[0])
    loss_type.set(values_nn[1])

    label10 = ttk.Label(options_frame, text="Training Method: ")
    label11 = ttk.Label(options_frame, text="Loss function: ")
    label12 = ttk.Label(options_frame, text="Learning Rate (advanced not required): ")

    label10.grid(row=0, column=0, padx=30)
    training_method.grid(row=0, column=1, padx=5, pady=5)
    label11.grid(row=1, column=0)
    loss_type.grid(row=1, column=1, padx=5, pady=5)
    # label12.grid(row=2, column=0)
    # learning_rate_nn.grid(row=2, column=1, padx=5, pady=5)
    options_frame.grid(row=1, pady=10)

    AddLayer(menu).grid()
    refresh_layer()
    show_layer()


def refresh_layer():
    global nn_layers
    global layers_list
    layers_list = []
    for layer in nn_layers:
        layers_list.append(Layer(menu, layer["neurons"], layer["activation"]))


def show_layer():
    for obj in layers_list:
        obj.grid()


def f(x, y, equation=lambda: get_equation()):
    return eval(equation)


def toggle_show_points():
    if use_or_show_data_points_bool.get():
        use_wireframe.grid(row=4)
        root.update()
        settings.update()

    elif not use_or_show_data_points_bool.get():
        use_wireframe.grid_forget()


def toggle_surface_menu():
    if wireframe_bool.get():
        style.grid(row=5)
        root.update()
        settings.update()

    elif not wireframe_bool.get():
        style.grid_forget()


def toggle_loadingbar():
    if use_or_not.get():
        label7.grid(row=13)
        style_nn.grid(row=14)
        change_nn.grid(padx=pad_x, pady=pad_y, row=15)
        # stats.grid()
        label6.grid(row=19)
        progress_bar.grid(row=20)
        root.update()
        settings.update()

    elif not use_or_not.get():
        label7.grid_forget()
        style_nn.grid_forget()
        change_nn.grid_forget()
        # stats.grid_forget()
        label6.grid_forget()
        progress_bar.grid_forget()


def get_cm(text):
    dict_cm = {"jet": cm.jet, "rainbow":cm.rainbow, "RdBu": cm.RdBu, "viridis": cm.viridis, "plasma": cm.plasma, "magma": cm.magma}
    return dict_cm[text] if text in dict_cm else cm.jet


def plot_data(amt_noise, size, res, equation, wireframe=False, points=True):
    x = np.arange(-size, size, res)
    y = np.arange(-size, size, res)
    x, y = np.meshgrid(x, y)
    z = f(x, y, equation=equation)
    noise = np.random.normal(-1, 1, z.shape) * amt_noise
    z -= noise
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    if not wireframe and points:
        ax.scatter(x, y, z)

    elif wireframe and points:
        global color_values
        ax.plot_surface(x, y, z, cmap=get_cm(color_values[style.current()]))

    plt.draw()

    return x, y, z, plt


def get_data(x, y, z):
    inputs = []
    outputs = []
    for i in range(len(x)):
        for j in range(len(x)):
            inputs.append([x[i][j], y[i][j]])
            outputs.append(z[i][j])

    data = np.array(inputs)

    labels = np.array(outputs)

    return data, labels


def make_model(data, labels):
    start_time = time.time()
    global nn_layers
    global values_nn
    model_layers = []

    string_text_output = "Layer {}, Number of Neurons in This Layer: {}, Activation: {}\n"
    for i, layer_val in enumerate(nn_layers):
        if i == 0:
            model_layers.append(layers.Dense(layer_val["neurons"], activation=layer_val["activation"], input_dim=2))
            write_log("\n" + string_text_output.format(i+1, layer_val["neurons"], layer_val["activation"]))
        elif i > 0:
            model_layers.append(layers.Dense(layer_val["neurons"], activation=layer_val["activation"]))
            write_log(string_text_output.format(i+1, layer_val["neurons"], layer_val["activation"]))

    model_layers.append(layers.Dense(1, activation="elu"))
    model = tf.keras.Sequential(model_layers)

    model.compile(optimizer=values_nn[0],
                  loss=values_nn[1],
                  metrics=['accuracy'])
    write_log("Optimizer: {}\nLoss Function: {}".format(values_nn[0], values_nn[1]))

    # Trains for 5 epochs
    history = model.fit(data, labels, batch_size=100, epochs=10, steps_per_epoch=200)
    write_log("\nTraining Took {} seconds\n".format(time.time() - start_time))
    #write_log("Final loss: {}\n\n".format(history.history["loss"][-1]))
    return model


def get_equation():
    code = entry_function.get()
    return code


def make_graph(x, y, model):
    inputs = []
    for i in range(len(x)):
        for j in range(len(x)):
            inputs.append([x[i][j], y[i][j]])

    inputs = np.array(inputs)
    predictions = []
    loss_list = []

    for i in range(len(inputs)):
        predictions.append(model.predict([[inputs[i]+resolution.get()/2]]))

    return np.array(predictions), inputs


def make_function(axis, x):
    buffer = 0
    half = []
    final = []
    for prediction in axis:
        if buffer < x:
            half.append(prediction[0][0])
            buffer += 1
        elif buffer >= x:
            final.append(half)
            half = []
            buffer = 0

    return np.array(final)


def go():
    """
    amt_moise = noise.get()
    size_loc = size.get()
    res = resolution.get()
    equation = get_equation()
    """
    amt_moise = 0.29605263157894735
    size_loc = 5
    res = 0.11960526315789473
    equation = "5*sin(x)+5*sin(y)+9.5"
    use_nn_bool = use_or_not.get()
    write_log("Amount of Noise: {}\nResolution: {}\nEquation: {}\nSize: {}\n\n". format(amt_moise, res, equation, size_loc))
    x, y, z, plot = plot_data(amt_moise, size_loc, res, equation, wireframe=wireframe_bool.get(), points=use_or_show_data_points_bool.get())
    if use_nn_bool == 1:
        label6.config(text="Status: Generating Data Points")
        progress_bar.step(20)
        root.update()

        data, labels = get_data(x, y, z)
        label6.config(text="Status: Training AI")
        progress_bar.step(50)
        root.update()

        model = make_model(data, labels)

        label6.config(text="Status: Making Graph From AI")
        progress_bar.step(25)
        root.update()

        z_pre, inputs = make_graph(x, y, model)
        ax = plot.gca(projection='3d')
        global color_values
        ax.plot_trisurf(x.flatten(), y.flatten(), z_pre.flatten(), cmap=get_cm(color_values[style_nn.current()]))
        data_t, labels_t = get_data(x, y, z)
        write_log("Total loss: {}\n---------------------------------\n\n".format(model.evaluate(data_t, labels_t)[0]))
        label6.config(text="Status: Done!")
        progress_bar.step(5)

        plot.show()

    else:
        plot.show()


nn_layers = [{"activation": "elu", "neurons": 64}, {"activation": "elu", "neurons": 128}, {"activation": "elu", "neurons": 64}]
values_nn = ["Adadelta", "logcosh"]
output_console = ""
color_values = ["rainbow", "RdBu", "viridis", "plasma", "magma", "jet", ]

root = tk.Tk()
root.title("Neural Network Demonstration")
settings = ttk.Frame(root)
options = ttk.Frame(root)
stats = ttk.Frame(root)

welcome = ttk.Label(settings, text="Welcome to the neural network demonstration!", font=('Helvetica', '20'))

label1 = ttk.Label(settings, text="Enter the function to mimic:")
entry_function = ttk.Entry(settings)
entry_function.insert(0, "x**2+y**2")

label2 = ttk.Label(settings, text="Select the amount of noise:")
noise = ttk.Scale(settings, from_=0, to=0.5, value=0.2)

label3 = ttk.Label(settings, text="Select the range of the function:")
size = ttk.Scale(settings, from_=0.1, to=5, value=1)

label4 = ttk.Label(settings, text="Select the resolution of the points:")
resolution = ttk.Scale(settings, from_=0.01, to=0.5, value=0.07)

use_or_not = tk.IntVar()
show_nn = ttk.Checkbutton(settings, text="Use neural network", variable=use_or_not, command=toggle_loadingbar)

change_nn = ttk.Button(settings, text="Configure Neural Network", command=nn_settings)

use_or_show_data_points_bool = tk.IntVar()
show_data_points = ttk.Checkbutton(settings, text="Show data points", variable=use_or_show_data_points_bool, command=toggle_show_points)

wireframe_bool = tk.IntVar()
use_wireframe = ttk.Checkbutton(settings, text="Use surface to show points (experimental) don't use", variable=wireframe_bool, command=toggle_surface_menu)

label5 = ttk.Label(settings, text="Enter the theme for graph:")
style = ttk.Combobox(settings, values=color_values, state='readonly')
style.set("jet")

label6 = ttk.Label(settings, text="Status: Not Started")
progress_bar = ttk.Progressbar(settings, mode="determinate", length=300)

label7 = ttk.Label(settings, text="Enter the theme for neural network graph:")
style_nn = ttk.Combobox(settings, values=color_values, state='readonly')
style_nn.set("jet")

go_button = ttk.Button(settings, text="GO!", command=go)

stats = ttk.Button(settings, text="Show Stats")

pad_x = 30
pad_y = 2

welcome.grid(padx=pad_x, pady=pad_y, row=0)
label1.grid(padx=pad_x, pady=pad_y, row=1)
entry_function.grid(padx=pad_x, pady=pad_y, row=2)
show_data_points.grid(padx=pad_x, pady=pad_y, row=3)
label2.grid(padx=pad_x, pady=pad_y, row=6)
noise.grid(padx=pad_x, pady=pad_y, row=7)
label3.grid(padx=pad_x, pady=pad_y, row=8)
size.grid(padx=pad_x, pady=pad_y, row=9)
label4.grid(padx=pad_x, pady=pad_y, row=10)
resolution.grid(padx=pad_x, pady=pad_y, row=11)
show_nn.grid(padx=pad_x, pady=pad_y, row=12)
go_button.grid(padx=pad_x, pady=pad_y, row=17)
settings.grid()

root.mainloop()
