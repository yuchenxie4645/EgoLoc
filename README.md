# EgoLoc: Zero-Shot Temporal Interaction Localization for Egocentric Videos

Authors: [Erhang Zhang*](https://scholar.google.com/citations?user=j1mUqHEAAAAJ&hl=en), [Junyi Ma*](https://github.com/BIT-MJY), [Yin-Dong Zheng](https://dblp.org/pid/249/8371.html), [Yixuan Zhou](https://ieeexplore.ieee.org/author/37089460430), [Hesheng Wang](https://scholar.google.com/citations?user=q6AY9XsAAAAJ&hl)

EgoLoc is a VLM-based paradigm to localize the hand-object contact/separation timestamps for egocentric videos in a zero-shot manner. Our [paper](https://arxiv.org/abs/2506.03662) is accepted by IROS 2025. We therefore extend the coarse temporal action localization (TAL) to finer-grained temporal interaction localization (TIL).

<div align="center">
  <span style="font-size: 20px; font-weight: bold;">
    from <span style="color: blue;">TAL</span> to <span style="color: red;">TIL</span>
  </span>
  <br />  <!-- 换行 -->
  <img src="TAL_TIL.png" alt="TAL to TIL Diagram" width="400" />
</div>

## 1. How to Use

We provide two demos from the [EgoPAT3D-DT dataset](https://github.com/oppo-us-research/USST) for quick use.

### 1.1 Install Dependencies 

Please install the dependencies of GroundedSAM following [this repo](https://github.com/IDEA-Research/Grounded-Segment-Anything). In this repo, we will show how to obtain 2D hand velocities to extract keyframes for temporal interaction localization. 

```bash
git clone https://github.com/IRMVLab/EgoLoc.git
cd EgoLoc
pip install -r requirements.txt

git clone https://github.com/IDEA-Research/Grounded-Segment-Anything
# Please install the dependencies of GroundedSAM
```

### 1.2 Run EgoLoc

We provide two exampled videos to show how EgoLoc works in a closed-loop manner. 

```bash
python egoloc_2D_demo.py --video_path ./video1.mp4 --output_dir output --config Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --grounded_checkpoint Grounded-Segment-Anything/groundingdino_swint_ogc.pth --sam_checkpoint Grounded-Segment-Anything/sam_vit_h_4b8939.pth --bert_base_uncased_path  Grounded-Segment-Anything/bert-base-uncased/ --text_prompt hand --box_threshold 0.3 --text_threshold 0.25 --device cuda --credentials auth.env --action "Grasping the object" --grid_size 3 --max_feedbacks 1
```
The output TIL results are saved in `output` folder.

|               | Video                | Contact                          | Separation                        |
|---------------|----------------------|----------------------------------|-----------------------------------|
| **video1**    | -                    | ![contact1](output/video1_contact_frame.png) | ![separation1](output/video1_separation_frame.png) |
| **video2**    | -                    | ![contact2](output/video2_contact_frame.png) | ![separation2](output/video2_separation_frame.png) |

Note that EgoLoc may output slightly different TIL results for different runs due to the inherent randomness in VLM-based reasoning.




### 1.3 Configurations

* **video_path**

* **output_dir**

* **config**

* **text_prompt**

* **box_threshold**

* **action**

* **grid_size**

* **max_feedbacks**


We will release the full-blood version of EgoLoc after we evolve to a more powerful solution.

## 2. Citation

🤝 If our work is helpful to your research, we would appreciate a citation to our paper:

```
@article{zhang2025zero,
  title={Zero-Shot Temporal Interaction Localization for Egocentric Videos},
  author={Zhang, Erhang and Ma, Junyi and Zheng, Yin-Dong and Zhou, Yixuan and Wang, Hesheng},
  journal={arXiv preprint arXiv:2506.03662},
  year={2025}
}
```


