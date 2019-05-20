# KitcheNette: Predicting and Recommending Food Ingredient Pairings using Siamese Neural Networks
This repository provides a Pytorch implementation of **KitcheNette**, Siamese neural networks and is trained on our annotated dataset containing 300K scores of pairings generated from numerous ingredients in food recipes. **KitcheNette** is able to predict and recommend complementary and novel food ingredients pairings at the same time.

> **KitcheNette: Predicting and Recommending Food Ingredient Pairings using Siamese Neural Networks** <br>
> *Donghyeon Park, Keonwoo Kim, Yonggyu Park, Jungwoon Shin and Jaewoo Kang* <br>
> *Accepted and to be appear in IJCAI-2019* <br><br>
> *Please, check our arxiv version of our paper:* <br>
> *http://arxiv.org/abs/1905.07261*

You can try our demo version of **KitchenNette**:
> *http://kitchenette.korea.ac.kr/*

### Currently, we are migrating the code and data. Some of the function may not work for now. Thank you for your patience.

## Abstract
As a vast number of ingredients exist in the culinary world, there are countless food ingredient pairings, but only a small number of pairings have been adopted by chefs and studied by food researchers. In this work, we propose KitcheNette which is a model that predicts food ingredient pairing scores and recommends optimal ingredient pairings. KitcheNette employs Siamese neural networks and is trained on our annotated dataset containing 300K scores of pairings generated from numerous ingredients in food recipes. As the results demonstrate, our model not only outperforms other baseline models but also can recommend complementary food pairings and discover novel ingredient pairings.

## Prerequisites
- Python 3.6
- PyTorch 0.4.0
- Numpy (>=1.12)
- Maybe there are more. If you get an error, please try `pip install "pacakge_name"`.

## Dataset
All the data are with code. They are in `data` folder.

### Currently, we are migrating the code and data. Some of the function may not work for now. Thank you for your patience.

## To run
Issue the command for ingredient embedding:
```
python main.py
```

## Contributors
Donghyeon Park, Keonwoo Kim

DMIS Labatory, Korea University, Seoul, South Korea

Please, report bugs and missing info to parkdh@korea.ac.kr.

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
