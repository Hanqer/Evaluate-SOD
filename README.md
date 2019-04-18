## Evaluation on saliency object detection(Evaluate SOD)
---
A **One-key** fast evaluation on saliency object detection with Muti-thread and GPU implementation including **MAE, Max F-measure, S-measure, E-measure**.

* Muti-thread in CPU
* GPU implementation with pytorch
* One-key evaluation

Usage:
```py
python main.py --root_dir 'your_dir' --save_dir 'your_dir' --methods 'DSS RAS' --dataset 'ECSSD SOD'    (if --methods and --dataset is not set, using all methods and datasets.)
```
**example:**
```py
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
        │   ├── 0001.jpg
        │   └── 0002.jpg
        ├── PASCAL-S
        │   ├── 1.jpg
        │   └── 2.jpg
        └── SOD
            ├── 2092.jpg
            └── 3096.jpg
```
TODO:
Add s-measure in two days.

