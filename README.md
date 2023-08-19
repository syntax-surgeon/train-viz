# train-viz

> Visualizing the training process of feed forward neural networks via class probability maps.<br>_(current version: 1.0)_
>
> #### Author: _Siddharth Yadav (syntax-surgeon)_<br><br>


## Files

### trainviz.py

- Contains version-1.0 of the train-viz program which provides functions to generate and visualize the training process via class probability maps

- The visualization relies on generating class probability maps for the model at different epochs during training

- The code follows a function approach in which both the generation and visualization of the probability maps are individual functions

- Currently supports only network models built in the PyTorch library (https://github.com/pytorch/pytorch)

- Please check the provided example.ipynb for a demonstration of the intended implementation

### example.ipynb

- Contains the demo example for the implementation of train-viz program

- The example utilizes a randomly generated seven class dataset (**Figure 1**) for the visualization process

- The dataset and the general network description is inspired by Mike X Cohen's course "Deep Understanding of Deep Learning" on Udemy.

> #### **Figure 1**: The seven class dataset used in the demo example 
>
> ![alt text](https://github.com/syntax-surgeon/train-viz/blob/main/readme_assets/7class_dataset.png?raw=true)
>

## Features

* **Mulitple visualization modes**<br>
The training process visualization can be modelled in two different modes (ref. _map_type_ parameter of **make_map** function):
    * Boundary mode - Emphasizes the boundary between the classes (**Figure 2**)

    * Region mode - Shows the region of the probability classes themselves (**Figure 3**)

    > #### **Figure 2**: Boundary mode visualization for seven class dataset 
    >
    > ![alt text](https://github.com/syntax-surgeon/train-viz/blob/main/readme_assets/boundary_7class_gif.gif?raw=true)

    > #### **Figure 3**: Region mode visualization for seven class dataset 
    >
    > ![alt text](https://github.com/syntax-surgeon/train-viz/blob/main/readme_assets/region_7class_gif.gif?raw=true)


* **Control over domain construction**<br>
The construction of the domain (input space) can be controlled dynamically or scaled (ref. _axial_gradation_ and _square_axis_points_ parameters of **make_map** funtion)

* **Map interpolation**<br>
Interpolation between maps can be achieved to increase the total number of maps. The rationale behind this is the following: To reduce memory utilization, class probability maps can be skipped for several epochs. However, this can cause "jagged" or "skipping" appearance of the animation. Interpolation can used to smoothen these artifacts. (ref. _interpolation_factor_ and _interpolation_type_ parameters of **plot_maps** function)

* **Simplified epoch skipping**<br>
To perform the aforementioned epoch skipping, additional logic will have to be implemented during the training process. This may be difficult to achieve in several situations. To circumvent this, the **make_map** function implements simple epoch skipping if the epoch number/index is provided. (ref. _epoch_num_ and _epoch_freq_ parameters of **make_map** function)

### Planned changes for the next version

* Additional customization of animation (eg: title, margin adjustment etc.)

* Add support for other deep learning libraries (eg: Tensorflow/Keras)

* Region mode can be improved with dedicated quantitative colormaps

* Support for developing static images

* Migrating to an object-oriented design