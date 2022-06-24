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
To recreate the figures present in the thesis the following Google Colab notebooks
are provided:
* Figure 4.2 : ` `
* Figures 4.3-4.4 and 4.6-4.11 : ` `
* Figure 5.1-5.2 and 5.4 : ` `
* Figure 5.3 : ` `

# Evaluation

**Mention that the ./models directory is copied from sefa + usage of interfacegan directions**
