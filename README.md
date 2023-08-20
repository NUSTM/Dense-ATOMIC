# Dense-ATOMIC: Towards Densely-connected ATOMIC with High Knowledge Coverage and Massive Multi-hop Paths

![avatar](./img/figure.png)

Resources and Codes for [Dense-ATOMIC: Towards Densely-connected ATOMIC with High Knowledge Coverage and Massive Multi-hop Paths](https://aclanthology.org/2023.acl-long.742.pdf)

## Bibtex

```
@inproceedings{DBLP:conf/acl/ShenWX23,
  author       = {Xiangqing Shen and
                  Siwei Wu and
                  Rui Xia},
  editor       = {Anna Rogers and
                  Jordan L. Boyd{-}Graber and
                  Naoaki Okazaki},
  title        = {Dense-ATOMIC: Towards Densely-connected {ATOMIC} with High Knowledge
                  Coverage and Massive Multi-hop Paths},
  booktitle    = {Proceedings of the 61st Annual Meeting of the Association for Computational
                  Linguistics (Volume 1: Long Papers), {ACL} 2023, Toronto, Canada,
                  July 9-14, 2023},
  pages        = {13292--13305},
  publisher    = {Association for Computational Linguistics},
  year         = {2023},
  url          = {https://aclanthology.org/2023.acl-long.742},
  timestamp    = {Thu, 13 Jul 2023 16:47:40 +0200},
  biburl       = {https://dblp.org/rec/conf/acl/ShenWX23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

<h1> Dense-ATOMIC </h1>

We currently release two versions of Dense-ATOMIC. More work is in progress.

Dense-ATOMIC-base:

[baidu disk](https://pan.baidu.com/s/1zQsX26MHTp3Hcxac5czsJw?pwd=nkbo  ), [google drive](https://drive.google.com/file/d/1yET0FAEej6LQtwBVVYR8nEgvFKn3xWUv/view?usp=drive_link )

| total number | 1153755 |
| ------------ | ------- |
| xNeed        | 157721  |
| xIntent      | 201780  |
| oAfter       | 224476  |
| xAfter       | 322034  |
| oPersona     | 91413   |
| xPersona     | 156331  |

Dense-ATOMIC-large:

[baidu disk](https://pan.baidu.com/s/1G3Lngc-J526Bishl5a-WYA?pwd=ld8x ), [google drive](https://drive.google.com/file/d/1AoTTJ3J-oGAtTi16xzonqPoSy2R_sy5-/view?usp=drive_link)

| total number | 10281235 |
| ------------ | -------- |
| xNeed        | 637624   |
| xIntent      | 1104854  |
| oAfter       | 1607797  |
| xAfter       | 1964070  |
| oPersona     | 2055146  |
| xPersona     | 2911744  |

<h1>Rel-CSKGC</h1>

The Dada for training and testing Rel-CSKGC can be download: [baidu_disk](https://pan.baidu.com/s/1unt6l6H7ZGuMcB1tm6HL0Q?pwd=6zi7) and [google_drive](https://drive.google.com/file/d/1_KnZ27tnp0IGce2fQJkoSUdNPHsEx6PL/view?usp=sharing).

Please unzip it under './Rel-CSKGC/' folder.

## Environment

- Python 3.6.9
- Cuda 11.0
- Run `pip install -r requirements.txt` to install the required packages.

## Training

We provide the Rek-CSKGC model here: [baidu_disk](https://pan.baidu.com/s/12YyE0CsSps7KUqFVVTBGyA?pwd=gtxh) and [google_drive](https://drive.google.com/file/d/1YDzVO3Z5V52o8x5zA3bDgkqmkX-BOshj/view?usp=sharing).

You can retrain the Rel-CSKGC model as following:

```bash
cd Rel-CSKGC
python run_training.py
```

## Testing

You can evaluate the Rel-CSKGC model on our human annotated testing dataset as following:

```bash
python run_predicting.py
```

## Creating Dense-ATOMIC

You can create the Dense-ATOMIC as following:

```bash
python run_completion.py
```

