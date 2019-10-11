## Evaluation on saliency object detection(Evaluate SOD)
---
A **One-key** fast evaluation on saliency object detection with GPU implementation including **MAE, Max F-measure, S-measure, E-measure**.

Code are reimplemented from the matlab version which are available from http://dpfan.net/

* GPU implementation with pytorch which can be easier embedding into eval code.
* One-key evaluation

Usage:
```
python main.py --root_dir 'your_dir' --save_dir 'your_dir' --methods 'DSS RAS' --dataset 'ECSSD SOD'    (if --methods and --dataset is not set, using all methods and datasets.)
```
**example:**
```
python main.py --root_dir './' --save_dir './'
```
example root_dir:
```
.
├── gt
│   ├── ECSSD
│   │   ├── 0001.png
│   │   └── 0002.png
│   ├── PASCAL-S
│   │   ├── 1.png
│   │   └── 2.png
│   └── SOD
│       ├── 2092.png
│       └── 3096.png
└── pred
    └── dss
        ├── ECSSD
        │   ├── 0001.png
        │   └── 0002.png
        ├── PASCAL-S
        │   ├── 1.png
        │   └── 2.png
        └── SOD
            ├── 2092.png
            └── 3096.png
```


If you find the code useful to your research, please cite the following papers.
```
@inproceedings{fan2018SOC,
	title={Salient Objects in Clutter: Bringing Salient Object Detection to the Foreground},
	author={Fan, Deng-Ping and Cheng, Ming-Ming and Liu, Jiang-Jiang and Gao, Shang-Hua and Hou, Qibin and Borji, Ali},
	booktitle = {European Conference on Computer Vision (ECCV)},
	year={2018},
	organization={Springer}
}


@inproceedings{fan2017structure,
	title={{Structure-measure: A New Way to Evaluate Foreground Maps}},
	author={Fan, Deng-Ping and Cheng, Ming-Ming and Liu, Yun and Li, Tao and Borji, Ali},
	booktitle={IEEE International Conference on Computer Vision (ICCV)},
	pages = {4548-4557},
	year={2017},
	note={\url{http://dpfan.net/smeasure/}},
	organization={IEEE}
}

@inproceedings{Fan2018Enhanced,
	author={Fan, Deng-Ping and Gong, Cheng and Cao, Yang and Ren, Bo and Cheng, Ming-Ming and Borji, Ali},
	title={{Enhanced-alignment Measure for Binary Foreground Map Evaluation}},
	booktitle={International Joint Conference on Artificial Intelligence (IJCAI)},
	pages={698--704},
	note={\url{http://dpfan.net/e-measure/}},
	year={2018}
}
```
