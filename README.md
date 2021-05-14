# Lagrangian Fluid Simulation with Continuous Convolutions and Timestep Prediction
## Introduction
This repository contains a modified version of [Lagrangian Fluid Simulation with Continuous Convolutions](https://github.com/intel-isl/DeepLagrangianFluids). We add a Timestep prediction component with the similar continuous convolution architecture into the original neural network, so that timestep used varies at different step during inference. The code allows you to generate training data (you will need [SPLishSPlash_with_time](https://github.com/horizon-research/SPlisHSPlasH_with_time.git)), train your own model or just run a pretrained model.

### Architecture
the architecture of original neural network can be illustrated as the following:
![image](https://github.com/horizon-research/DeepFluid-with-Timestep-prediction/blob/master/readmeimages/original_archi.png)

our modified architecture keeps the manner but makes timestep variable
![image](https://github.com/horizon-research/DeepFluid-with-Timestep-prediction/blob/master/readmeimages/our_archi.png)

Please refer to [Lagrangian Fluid Simulation with Continuous Convolutions](https://github.com/intel-isl/DeepLagrangianFluids) for installing dependencies and basic usage.
______
## Running the pretrained model
```
./run_network.py --weights model_weights.pt \
                 --scene upper.json \
                 --output example_out \
                 --write-ply \
                 train_PosCorrectionNetwork.py
```
____
### Tips on initialization scene
The .json file given to --scene parameter defines the initial scene for inference, in which we define the initial position and shape of *wall* and *fluid*. Both of them can be defined by .obj file or .bgeo file. If you write a path to .obj file, you need to call the **VolumeSampling tool of SPLishSPlash** in *run_network.py* to transform it into particles data. If you write a path to .bgeo file, since it is already particles data, the only thing needed is to read it out in *run_network.py*.

Code involved:
```
fluids = []
for x in scene['fluids']:
    # points = obj_volume_to_particles(x['path'])[0]
    points = numpy_from_bgeo(x['path'])[0]
```
____
## Training the network
### Data generation 

The data generation scripts are in *datasets/create_data.sh*. Since the typical process of using SPlisHSPlasH is inputing a .json file, then SPHSimulator will run a simulation based on settings defined in the .json file and the result of simulation will be output, *create_physics_scenes.py* corresponds to the .json file. This code aims to generate different settings (.json files). Each setting will be used once in SPHSimulator, thus different results generated. 

Note that our current *create_physics_scenes.py* corresponds to **SPlisHSPlasH v2.9.0**, if it has been updated, you need to modify this code according to their lateset documentation, or according to examples in the *data/Scene/*.json* of the lateset SPlisHSPlasH. 

Then these different results will be read by *create_physics_records.py* and transformed into a form suitable for our neural network. As we add time data in exporter of SPHSimulator, so it has been changed accordingly. Also, the *dataset_reader_physics.py* is the dataloader of our neural network, it also has been changed accordingly, so that it can yield sample with time data. 

1. set the path to the *SPHSimulator* of SPlisHSPlasH in the *datasets/splishsplash_config.py*. Generally, SPHSimulator is in the *SPlisHSPlasH/bin* after SPlisHSPlasH installed. If you want to generate dataset with time data, please set the path to the *SPHSimulator* of our [SPlisHSPlasH_with_time](https://github.com/horizon-research/SPlisHSPlasH_with_time.git). Otherwise, please set to the SPHSimulator of original SPlisHSPlasH.
2. run the scripts from within the datasets folder
```
cd datasets
./create_data.sh
```
____
### Training scripts
Our neural network has two components: TimeStep Prediction component and Position Correction Prediction component. We train the timestep prediction network first, then we load the pretrained weight into this component, when we initialize our position correction prediciton network. Then position correction prediction network is trained with the same train set but use different data in it. Of course, you can try other better training strategies.

```
cd scripts
# train timestep prediction network
./train_TimestepPreNetwork.py
```
```
cd scripts
# train position correction prediction network with pretrained weights for timestep prediction component
./train_PosCorrectionNetwork.py --ts_weight ts_weights.pt
```
____
## Rendering your result
You could refer to [this](https://github.com/intel-isl/DeepLagrangianFluids/blob/master/scenes/README.md). In the example provided by the author, we can render the canyon scene. If you want to render you own scene, you can refer to these tips:  

1. **Import geometry file.** You can import fluid.obj into blender by: file -> import. Under the object properties tab, you can add Customer properties. The most import one is external_files. Put the path to the .npz files or .ply files you get after run the model. This property will be used by the script, so make sure you keep the name of the property consistent.  

2. **Script to load data for an object.** *blender_external_mesh.py* is provided by the author and you can use it to load the particle data into the FluidParticles every time the current frame changes.   

3. **Create your own object.** You may also want to build a complex scene as the canyon one. Here is [a good tutorial](https://www.youtube.com/playlist?list=PL3UWN2F2M2C8-zUjbFlbgtWPQa0NXBsp0) we found to create an object in Blender. Other sources can be found in our [slide](https://docs.google.com/presentation/d/1aJVyZiPywuSXqridX2u4AhRl_u_tFVOwjzDEp6grHoM/edit#slide=id.gd29dfda11b_1_0).  

____
## Utilities & Tips
1. *read_ckpt.py* is used to read weights from checkpoints. You might need it when you want to check about the performance of some intermediate checkpoints. 
```
cd utils/others
./read_ckpt.py --ckpt ckpt-1000.pt --output model_weight.pt --modeltype pos|ts
```
2. *animation.py* is used to visualize the result of prediction. After your *run_network.py* finishes, you will get a folder containing all the results. Just input the path to the folder as the parameter of the function then run the code.

3. Please refer to the [original neural network](https://github.com/intel-isl/DeepLagrangianFluids) for other things like rendering particles into fluid, network evaluation and more details.
____
## To-do list
1. GPU consumption may accumulate during training, you could use a smaller batch size for now
2. Redesign the loss function according to your training strategy. Currently, since timestep prediction component is trained with absolute difference between predicted timestep and ground truth. During inference, timestep prediction component may predict negative value, which absolutely has bad influence on position correction prediction.
___
## Materials
[our previous meetings](https://drive.google.com/drive/folders/1fcvJHtZU4Joo9z6-J8NeL-u6gXXDSKb8?usp=sharing)

[corresponding videos](https://drive.google.com/drive/folders/1tZnBEz900CDmCtcEg5tLgE47LYRQDMlg?usp=sharing) 
___
## Contacts
Please feel free to contact us for any questions

hanlingao艾特outlook.com

zeyipan艾特hotmail.com
