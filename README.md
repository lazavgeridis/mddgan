# Introduction

This repo contains the implementation of my B.sc. Thesis titled
**"MddGAN : Multilinear Analysis of the GAN Latent Space"**. The thesis
text can be found [here](https://pergamos.lib.uoa.gr/uoa/dl/object/3059772).

In short, this thesis proposes an unsupervised method to discover a wide range
of interpretable vector directions by analyzing the space of the generator's
parameters, otherwise known as the GAN latent space. The extracted directions
can then be exploited in order to produce impressive visual edits, on par with
the current sota methods. Furthermore, the proposed method does not only reveal
the explanatory factors learnt by the generator, but it can also arrange them
along the dimensions of the produced multilinear basis, according to the ...
they ... .


# Results

**StyleGAN2 FFHQ**

![stylegan2_ffhq](images/stylegan2ffhq_chart.jpg)


**StyleGAN AnimeFaces**

![stylegan_animeface](images/stylegananime_chart.jpg)



# Usage
**User should install requirements file first.**

## Discovering semantic concepts in the GAN latent space
### Basic Execution
```
python discover_semantics.py [model_name] [method_name]
```
where `model_name` refers to the name of the GAN model you want to discover
semantics for and `method_name` refers to the method to use when analyzing
the latent space of the selected GAN model. The list of valid `model_name`'s
to use can be found at [mddgan/models/model_zoo.py](models/model_zoo.py),
while `method_name` can be either one of `mddgan`, `sefa` or `both`.

For instance, some sample executions are:

```
# Analyze StyleGAN2 FFHQ model
python discover_semantics.py stylegan2_ffhq1024 [method_name]

# Analyze StyleGAN LSUN Bedroom model
python discover_semantics.py stylegan_bedroom256 [method_name]

# Analyze ProGAN CelebaHQ model
python discover_semantics.py pggan_celebahq1024 [method_name]
```

### Analyzing Specific Layer Ranges
Note that in the case of **StyleGAN/StyleGAN2** models, e.g `stylegan2_ffhq1024` and
`stylegan_bedroom256` from above, the default behaviour of the program is to analyze
_all layers_ of the selected model, which will discover directions
that impact multiple variation factors at once. However, this behaviour can be modified by
using the `layer_range` option. For example, to extract semantics that effect the overall
geometric properties of the image, you probably want to target the initial layers:

```
python discover_semantics.py stylegan2_car512 [method_name] --layer_range 0-3
```
In general, the argument to `layer_range` indicates the layer indices of
the model to analyze and is of the form: $idx_{1} - idx_{2}$, where
$idx \in [0, L]$ (L is the total number of layers in $G$).

### Attempting to Group the Discovered Semantics
Other than simply discovering surprising directions, MddGAN can additionally
separate them into groups. In essence, by tensorizing the produced multilinear
basis $\mathcal{\mathbf{B}}$, one can attempt to gather all directions encoding
the same variability factor by slicing tensor $\mathcal{\mathbf{B}}$ on the
appropriate mode. To achieve this, we can use the `num_modes` option. The
argument to `num_modes` sets the estimated number of variation factors the
Generator has learnt to model. For instance, assuming 3 modes of variation:

```
python discover_semantics.py stylegan2_car512 mddgan --num_modes 3
```

### Reducing the Number of Discovered Directions
Finally, to discover a reduced number of directions, the `num_components`
option can be used. For instance, to discover 200 directions instead of
the default 512, run:

```
python discover_semantics.py stylegan2_car512 mddgan --num_components 200
```

### Selecting the Editing Magnitude
talk about the magnitude of the edit `start_distance` and `end_distance`.

## Evaluation
### FID Scores
In the directory [mddgan/fid_files](fid_files), we provide some pre-computed
FID scores for some distinctive facial attributes (pose, gender, age, smile,
eyeglasses).

For example, to plot the FID scores for the pose discovered attribute and for
the StyleGAN CelebaHQ model, comparing MddGAN to InterFaceGAN, run:

```
python plot_fid.py stylegan_celebahq1024 interfacegan pose
```

The program will locate the corresponding file, in this case the file is 
[mddgan/fid_files/stylegan_celebahq1024_interfacegan_pose.txt](fid_files/stylegan_celebahq1024_interfacegan_pose.txt),
gather the FID scores and produce the corresponding plot.

### Correlation Between Discovered Attributes 
cover `cosine_similarity.py`

## Reproducibility
* Mention that the code of this repo requires a machine with GPU to run. If the reader doesn't
  have a GPU available, he/she can still run the notebooks.

To recreate the figures present in the thesis the following Google Colab notebooks
are provided:
* Figure 4.2 : ` `
* Figures 4.3-4.4 and 4.6-4.11 : ` `
* Figure 5.1-5.2 and 5.4 : ` `
* Figure 5.3 : ` `



# Acknowledgements
This project could not exist if it weren't for the excellent implementations
mentioned below:
* The [SeFa](https://github.com/genforce/sefa) project, from which a substantial
part of the code of this project is inspired. The [mddgan/models](models)
directory used here is borrowed from SeFa.
* The [InterFaceGAN](https://github.com/genforce/interfacegan) project, from
which we borrow the ProGAN and StyleGAN directions used in our comparisons.
* The [GANLatentDiscovery](https://github.com/anvoynov/GANLatentDiscovery)
project, from which we got the inspiration for the core visualization operation
implemented here.

**Mention that the ./models directory is copied from sefa + usage of interfacegan directions**
