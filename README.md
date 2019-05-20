# KitcheNette: Predicting and Recommending Food Ingredient Pairings using Siamese Neural Networks
This repository provides a Pytorch implementation of **KitcheNette**, Siamese neural networks and is trained on our annotated dataset containing 300K scores of pairings generated from numerous ingredients in food recipes. **KitcheNette** is able to predict and recommend complementary and novel food ingredients pairings at the same time.

> **KitcheNette: Predicting and Recommending Food Ingredient Pairings using Siamese Neural Networks** <br>
> *Donghyeon Park\*, Keonwoo Kim, Yonggyu Park, Jungwoon Shin and Jaewoo Kang* <br>
> *Accepted and to be appear in IJCAI-2019* <br><br>
> *arxiv version of our paper is available at:* <br>
> *http://arxiv.org/abs/1905.07261* <br><br>
> You can try our demo version of **KitchenNette**: <br>
> *http://kitchenette.korea.ac.kr/*

## Pipeline & Abstract
![figure](/data/figure_together.png)
<p align="center">
  <b> The Concept of KitcheNette (Left) & KitcheNette Model Architecture (Right) </b>
</p>

**Abstract** <br>
As a vast number of ingredients exist in the culinary world, there are countless food ingredient pairings, but only a small number of pairings have been adopted by chefs and studied by food researchers. In this work, we propose KitcheNette which is a model that predicts food ingredient pairing scores and recommends optimal ingredient pairings. KitcheNette employs Siamese neural networks and is trained on our annotated dataset containing 300K scores of pairings generated from numerous ingredients in food recipes. As the results demonstrate, our model not only outperforms other baseline models but also can recommend complementary food pairings and discover novel ingredient pairings.

## Prerequisites
- Python 3.6
- PyTorch 0.4.0
- Numpy (>=1.12)
- Maybe there are more. If you get an error, please try `pip install "pacakge_name"`.

## Dataset
- **[kitchenette_pairing_scores.csv](https://drive.google.com/open?id=1hX7L3UZUVspNHCjDbgCjuI5niQlBXXMh) (78MB)** <br>
You can download and see our 300k food ingredient pairing scores defined on NPMI.

- **\[For Training\] [kitchenette_dataset.pkl](https://drive.google.com/open?id=1tUbwr7COW0lkiGkM3gafeGwtQncWd8wC) (49MB)** <br>
For your own training, download our pre-processed Dataset (49MB) and place it in `data` folder. This pre-processed dataset 1) contains all the input embeddings, 2) is split into train(8):valid(1):test(2), and 3) and each split is divided into mini-batches for efficent training.

## Training & Test
```
python3 main.py --data-path './data/kitchenette_dataset.pkl'
```
## Prediction for *Unknown* Pairings
- **\[For Prediction\] [kitchenette_pretrained.mdl](https://drive.google.com/open?id=1y5lFnECVdAaEikezeYipIABo4-5gvcbb) (79MB)** <br>
Download our pre-trained model for prediction of *unknown* pairings <br>
or you can predict the pairing with your own model by substituting the model file.

```
python3 main.py --save-prediction-unknowns True --data-path './data/kitchenette_dataset.pkl'
```

## Contributors
Donghyeon Park, Keonwoo Kim

DMIS Labatory, Korea University, Seoul, South Korea

Please, report bugs and missing info to Donghyeon `parkdh (at) korea.ac.kr`.

## Citation

For now, cite *[arxiv](http://arxiv.org/abs/1905.07261)* version of our paper:

```
@article{park2019kitchenette,
  title={KitcheNette: Predicting and Recommending Food Ingredient Pairings using Siamese Neural Networks},
  author={Park, Donghyeon and Kim, Keonwoo and Park, Yonggyu and Shin, Jungwoon and Kang, Jaewoo},
  journal={arXiv preprint arXiv:1905.07261},
  year={2019}
}
```

## Liscense
Apache License 2.0
