## Full Self Navigation
A project built based on Surroundocc and POP3D aimed at predicting the occupancy of the environment based on 3 surrounding cameras input. OCC prediction is popular when Tesla announced its Full Self Driving system. Thus this project is inspired by it and aims to achieve similar results on Robot Vacuums product.

This project tries to solve this problem in 2 paradigms:
1. Manully label the voxels as GroundTruth labels and supervise the OCC prediction on voxel level.
2. Utilizing a finetuned FCCLIP to get semantics of  all the 2d images and supervise the OCC prediction on image level.


### Installation
The installation steps can be found in [install.md](docs/install.md)
### Open Vocalbulary segmentation (Paradigm 2)
The Open vocalbulary segmentation model is finetuned on tens of thousands of Robot Vacuums images and thus provide good open vocalbulary segmentation results in indoor scenes.

### Data
The data can be accquired on [HuggingFace]()
### Training and Evaluation
The training and evaluation step are describe in [run.md](docs/run.md)

Great thanks to following great projects：
- [FCCLIP](https://github.com/bytedance/fc-clip)
- [Surroundocc](https://github.com/weiyithu/SurroundOcc)
- [POP3D](https://vobecant.github.io/POP3D/)
