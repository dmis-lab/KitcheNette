# KitcheNette: Predicting and Recommending Food Ingredient Pairings using Siamese Neural Networks

A Pytorch Implementation of paper
> KitcheNette: Predicting and Recommending Food Ingredient Pairings using Siamese Neural Networks <br>
> Park et al., 2019
> Accepted and to be appear in IJCAI-2019
> arxiv: in process

Currently, we are migrating the code and data. 
Some of the function may not work for now.
Thank you for your patience.

## Abstract
As a vast number of ingredients exist in the culinary world, there are countless food ingredient pairings, but only a small number of pairings have been adopted by chefs and studied by food researchers. In this work, we propose KitcheNette which is a model that predicts food ingredient pairing scores and recommends optimal ingredient pairings. KitcheNette employs Siamese neural networks and is trained on our annotated dataset containing 300K scores of pairings generated from numerous ingredients in food recipes. As the results demonstrate, our model not only outperforms other baseline models but also can recommend complementary food pairings and discover novel ingredient pairings.

## Prerequisites
- Python 3.6
- PyTorch 0.4.0
- Numpy (>=1.12)
- Maybe there are more. If you get an error, please try `pip install "pacakge_name"`.

## Dataset
All the data are with code. They are in `data` folder.

Currently, we are migrating the code and data. 
Some of the function may not work for now.
Thank you for your patience.

 ## To run
Issue the command for ingredient embedding:
```
python run.py
```
 To set dataset specific hyperparameters modify `Config.py`.
 ## Contributors
Donghyeon Park, Yonggyu Park
Korea University
 Report bugs and missing info to dhyeon.park@gmail.com.


## Liscense
Apache License 2.0
