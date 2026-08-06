# TODO:

## Overview

We need to see whether a Multi-modal and intermediate fusion model will work better than a sequential model that feeds data into each other one by one.

Sequential:
CT/PET -> Segementation UNet -> Output 1
Output 1 + Clinical Data -> Random Forest Classification -> Output 2
Output 2 + Clinical Data -> Random Forest Regression -> Output 3

Multimodal Method:
CT -> Encoder Output 1
PET -> Encoder -> Model -> Output 2
Clinical -> Encoder Output 3

## How To Get This Shit Done Cause Like What

So like, we have three very separate inputs that enter one model for three different outputs. We can't just add them in without any preprocessing (acc we can; its called early fusion but its stupid), so instead we use a method called **intermediate fusion**.

Basically, before they enter the shared model, each input enters its own little mini model called an encoder.

CT Encoder: 3D Medical Image -> 3D CNN
PET Encoder: 3D Medical Image -> 3D CNN
Clinical Encoder: 1D Feature Vector -> Basic MLP (MultiPlayer Perceptron)

Then, we concatenate them into different layers (I think?) and then running another ML model, and then getting the outputs)
