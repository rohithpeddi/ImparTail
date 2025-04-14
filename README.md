<h1 align="center">
  🚀✨ Towards Unbiased and Robust Spatio-Temporal Scene Graph Generation and Anticipation ✨🚀
</h1>

<p align="center">  
  🌟 Rohith Peddi, Saurabh, Ayush, Parag Singla, Vibhav Gogate 🌟
</p>

<p align="center">
  🎤 CVPR 2025 (Highlight Presentation) 🎤
</p>

<div align="center">
  <a src="https://img.shields.io/badge/project-website-green" href="https://rohithpeddi.github.io/#/impartail">
    <img src="https://img.shields.io/badge/project-website-green">
  </a>
  <a src="https://img.shields.io/badge/paper-arxiv-red" href="https://arxiv.org/pdf/2403.04899v1.pdf">
    <img src="https://img.shields.io/badge/paper-arxiv-red">
  </a>
  <a src="https://img.shields.io/badge/bibtex-citation-blue" href="">
    <img src="https://img.shields.io/badge/bibtex-citation-blue">
  </a> 
</div>

<p align="center">
  🌈 (This page is under continuous update) 🌈
</p>

----

## 🔄 UPDATE

- **Feb 2025** - 🎊 Our paper is accepted at CVPR 2025 🎊
- **March 2025** - 🚀 Initial Release of Code and Checkpoints 🚀
- **Apr 2025** - 🌟 Our paper is selected as a highlight (Acceptance rate: **~3.7%** (387/13000)) 🌟


-------
### ACKNOWLEDGEMENTS

This code is based on the following awesome repositories. 
We thank all the authors for releasing their code. 

1. [STTran](https://github.com/yrcong/STTran)
2. [DSG-DETR](https://github.com/Shengyu-Feng/DSG-DETR)
3. [Tempura](https://github.com/sayaknag/unbiasedSGG)
4. [TorchDiffEq](https://github.com/rtqichen/torchdiffeq)
5. [TorchDyn](https://github.com/DiffEqML/torchdyn)
6. [SceneSayer](https://github.com/rohithpeddi/SceneSayer)


-------
# SETUP

## Dataset Preparation 

**Estimated time: 10 hours**

Follow the instructions from [here](https://github.com/JingweiJ/ActionGenome)

Download Charades videos ```data/ag/videos```

Download all action genome annotations ```data/ag/annotations```

Dump all frames ```data/ag/frames```

#### Change the corresponding data file paths in ```datasets/action_genome/tools/dump_frames.py```


Download object_bbox_and_relationship_filtersmall.pkl from [here](https://drive.google.com/file/d/19BkAwjCw5ByyGyZjFo174Oc3Ud56fkaT/view)
and place it in the data loader folder

### Install required libraries

```
conda create -n impartail python=3.7 pip
conda activate impartail
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install -r requirements.txt
```

### Build draw_rectangles modules

```
cd lib/draw_rectangles
```
Remove any previous builds
```
rm -rf build/
rm -rf *.so
rm -rf *.c
rm -rf *.pyd
```
Build the module
```
python setup.py build_ext --inplace
cd ..
```
Add the path to the current directory to the PYTHONPATH

```
conda develop draw_rectangles/
```

### Build bbox modules

```
cd fpn/box_intersections_cpu
```
Remove any previous builds
```
rm -rf build/
rm -rf *.so
rm -rf *.c
rm -rf *.pyd
```
Build the module
```
python setup.py build_ext --inplace
cd ..
```
Add the path to the current directory to the PYTHONPATH

```
conda develop fpn/box_intersections_cpu/
```

# fasterRCNN model

Remove any previous builds

``` 
cd fasterRCNN/lib
rm -rf build/
```

Change the folder paths in 'fasterRCNN/lib/faster_rcnn.egg.info/SOURCES.txt' to the current directory

```
python setup.py build develop
```

If there are any errors, check gcc version ``` Works for 9.x.x```


Follow [this](https://www.youtube.com/watch?v=aai42Qp6L28) for changing gcc version


Download pretrained fasterRCNN model [here](https://utdallas.box.com/s/1pspm5x8etlczoklyw4bclotsmlejagj) and place in fasterRCNN/models/

Download the pkl file from [here](https://utdallas.box.com/s/wioo5obxkggs7lyqftvp64fkbs2uz4eu) and place it in dataloader/


------

## CHECKPOINTS

We provide the checkpoints for the following methods:

### Video Scene Graph Generation

#### A) Fully Supervised Setting:

We trained each model for 5 epochs and provided the checkpoint that performed best on the MeanRecall Metrics on the 5 epochs. 

1. **DSG-DETR** 
2. **STTran**

Please find the checkpoints here: [LINK](https://utdallas.box.com/s/ji6aekd4d8iguqk59rfwcalhpu608mop)

#### B) ImparTail Trained Models:

We provide the checkpoints for the following methods:

1. **DSG-DETR**
2. **STTran**

Under the following settings:
1. Curriculum which is capped at a Maximum Mask Ratio of 0.3, 0.6, 0.9
2. For the following modes: sgdet, sgcls, predcls

Please find the checkpoints here: [LINK](https://utdallas.box.com/s/1515xwjjt04tnby8zfy9m8jnk9pztnv4)

### Video Scene Graph Anticipation

#### A) Fully Supervised Setting:

We trained each model for 5 epochs and provided the checkpoint that performed best on the MeanRecall Metrics on the 5 epochs.
We primarily focus on the future prediction of 3 frames during training as described in SceneSayer.

1. **DsgDETR+**
2. **STTran+**
3. **DsgDETR++** (DsgDETR-GEN-ANT)
4. **STTran++** (sttran_gen_ant)
5. **SceneSayerODE**
6. **SceneSayerSDE**

Please find the checkpoints here: [LINK](https://utdallas.box.com/s/ddr2wnwt0qi8ihrw69zdpo5o3blesukl)

#### B) ImparTail Trained Models:

We provide the checkpoints for the following methods:

1. **DsgDETR+**
2. **STTran+**
3. **DsgDETR++** (DsgDETR-GEN-ANT)
4. **STTran++** (sttran_gen_ant)
5. **SceneSayerODE**
6. **SceneSayerSDE**

Under the following settings:
1. Curriculum which is capped at a Maximum Mask Ratio of (0.3, 0.6, 0.9) in the curriculum. 
2. For the following modes: ags, pgags, gags

**NOTE**: Although the names follow the convention: sgdet(ags), sgcls(pgags), predcls(gags).

Please find the checkpoints here: [LINK](https://utdallas.box.com/s/exbgy57fjnkt5aahoz9z07njtw1y2z9o)

The checkpoints follow this naming structure.

```
<CKPT_METHOD_NAME>_[partial_((1-<MASK_RATIO>)*100)]_<MODE>_future_<#TRAIN_FUTURE_FRAMES>_epoch_<#STORED_EPOCH>.tar
```

Eg:

Mask Ratio = 0.3 --> **dsgdetr_ant**\_**partial**\_**70**\_**sgdet**\_future\_**3**\_epoch\_**0**

Mask Ratio = 0.9 --> **ode**\_**partial**\_**10**\_**sgdet**\_future\_**1**\_epoch\_**0**

------

# Instructions to run

Please see the scripts/training for Python modules.

Please see the scripts/tests for testing Python modules.

------


## Citation

If you find this code useful, please consider citing our paper:

```
@misc{peddi2024unbiasedrobustspatiotemporalscene,
      title={Towards Unbiased and Robust Spatio-Temporal Scene Graph Generation and Anticipation}, 
      author={Rohith Peddi and Saurabh and Ayush Abhay Shrivastava and Parag Singla and Vibhav Gogate},
      year={2024},
      eprint={2411.13059},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.13059}, 
}
```

