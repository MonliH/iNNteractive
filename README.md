# iNNteractive
Interactive neural network with a GUI interface. Programmed in python 3.6

The neural network was implemented using Google’s Tensorflow API and the Keras API. The plotting was done through the Matplotlib module for Python. The builtin tkinter module was used to create the user interface. The Numpy module was used to generate the training data. The python file itself is about 400 lines long. This program will generate a neural netowrk to fit any function that you give it.

# Features
<table>
  <tr>
    <td width="50%" valign="top">
<h3>Data Generation</h3>
Using Numpy, the training data can be automaticaly generated given a base function. The function takes the input x and y, then manipulates the values. On the right is the graph of `x**2+y**2`. THe range and the resolution of the function can be ajusted using user-friendly sliders.
    </td>
    <td>
      <img src="images/screenshot3.png">
    </td>
  </tr>
  <tr>
    <td width="50%">
      <img src="images/screenshot4.png">
    </td>
    <td valign="top">
<h3>Flexibility</h3>
The Numpy library gives this program a large amount of flexibility. Functions including sin, cos, and tan are supported. Advanced users can change the code with relative ease.
    </td>
  </tr>
<tr>
    <td width="50%" valign="top">
<h3>Create Noise In Training Data</h3>
Creating noise in training data allows more reliable and realistic results. The amount of noise can be specified.
    </td>
    <td>
      <img src="images/screenshot5.png">
    </td>
  </tr>
  <tr>
    <td width="50%">
      <img src="images/screenshot6.png">
    </td>
    <td valign="top">
<h3>Overlay Training Data With Neural Netowrk Prediction</h3>
This feature allows the output from the neural network and the real function to be compared easily.
    </td>
  </tr>

<tr>
    <td width="50%" valign="top">
<h3>Control The Neural Network</h3>
With some Keras magic, the structure of the neural network, the activation function, its training method, and its error function can be changed and set. This opens up many possibilities during training.
    </td>
    <td>
      <img src="images/screenshot2.png">
    </td>
  </tr>
  <tr>
    <td width="50%">
      <img src="images/screenshot1.png">
    </td>
    <td valign="top">
<h3>Change Theme Of Interactive Graph</h3>
With the power of MatplotLib, It is possiable to change the color theme of the interactive 3D graph. This feature is useful for comparing the output of the neural netowrk to the real function given.
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
<h3>Built On Tools You Trust!</h3>
Programmed in python, Tensorflow made by Google is the brain behind the neural network and is trusted by millions around the globe. Other libaries used include: Numpy, Matplotlib, and Tkinter.
    </td>
    <td>
      <img src="https://www.tensorflow.org/images/tf_logo_social.png">
    </td>
  </tr>
</table>


Other Features:
* An interactive graph to visualize the training data and the neural network output
* Status bar for training
* Log files (loss, structure of neural network, time etc.)


I suggest you play around with the program as it is quite intivtive to use. The function syntax is the normal python syntax (e.g. power is `**` and there is `sin(x)`, `cos(x)`, `tan(x)`).

# Dependencies
* Python 3
* Matplotlib
* Tensorflow
* Keras
* Numpy

To install the python modules run:

```
> pip install matplotlib

> pip install tensorflow

> pip install keras

> pip install numpy
```

In your terminal if on windows, make sure you have Python 3 put in your path. If on any unix os, use `pip3` instead of `pip`

Not tested on windows. Desgned for MacOS. Theoretically it should work on windows but im not sure.

