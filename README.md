# Strong baseline for visual question answering

This is a re-implementation of Vahid Kazemi and Ali Elqursh's paper [Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering][0] in [PyTorch][1].

The paper shows that with a relatively simple model, using only common building blocks in Deep Learning, you can get better accuracies than the majority of previously published work on the popular [VQA v1][2] dataset.

This repository is intended to provide a straightforward implementation of the paper for other researchers to build on.
The results closely match the reported results, as the majority of details should be exactly the same as the paper. (Thanks to the authors for answering my questions about some details!)
This implementation seems to consistently converge to about 0.1% better results –
there are two main implementation differences:

- Instead of setting a limit on the maximum number of words per question and cutting off all words beyond this limit, this code uses per-example dynamic unrolling of the language model.
- [An issue with the official evaluation code](https://github.com/Cyanogenoid/pytorch-vqa/issues/5) makes some questions unanswerable. This code does not normalize machine-given answers, which avoids this problem. As the vast majority of questions are not affected by this issue, it's very unlikely that this will have any significant impact on accuracy.

A fully trained model (convergence shown below) is [available for download][5].

![Graph of convergence of implementation versus paper results](http://i.imgur.com/moWYEm8.png)

Note that the model in [my other VQA repo](https://github.com/Cyanogenoid/vqa-counting) performs better than the model implemented here.


## Running the model

- Clone this repository with:
```
git clone https://github.com/Cyanogenoid/pytorch-vqa --recursive
```
- Set the paths to your downloaded [questions, answers, and MS COCO images][4] in `config.py`.
  - `qa_path` should contain the files `OpenEnded_mscoco_train2014_questions.json`, `OpenEnded_mscoco_val2014_questions.json`, `mscoco_train2014_annotations.json`, `mscoco_val2014_annotations.json`.
  - `train_path`, `val_path`, `test_path` should contain the train, validation, and test `.jpg` images respectively.
- Pre-process images (93 GiB of free disk space required for f16 accuracy) with [ResNet152 weights ported from Caffe][3] and vocabularies for questions and answers with:
```
python preprocess-images.py
python preprocess-vocab.py
```
- Train the model in `model.py` with:
```
python train.py
```
This will alternate between one epoch of training on the train split and one epoch of validation on the validation split while printing the current training progress to stdout and saving logs in the `logs` directory.
The logs contain the name of the model, training statistics, contents of `config.py`,  model weights, evaluation information (per-question answer and accuracy), and question and answer vocabularies.
- During training (which takes a while), plot the training progress with:
```
python view-log.py <path to .pth log>
```


## Python 3 dependencies (tested on Python 3.6.2)

- torch
- torchvision
- h5py
- tqdm



[0]: https://arxiv.org/abs/1704.03162
[1]: https://github.com/pytorch/pytorch
[2]: http://visualqa.org/
[3]: https://github.com/ruotianluo/pytorch-resnet
[4]: http://visualqa.org/vqa_v1_download.html
[5]: https://github.com/Cyanogenoid/pytorch-vqa/releases
