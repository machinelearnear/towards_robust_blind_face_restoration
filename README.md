# Towards Blind Face Restoration

From their [official website](https://auto.gluon.ai/stable/index.html):
> AutoGluon enables easy-to-use and easy-to-extend AutoML with a focus on automated stack ensembling, deep learning, and real-world applications spanning image, text, and tabular data. Intended for both ML beginners and experts, AutoGluon enables you to:
> - Quickly prototype deep learning and classical ML solutions for your raw data with a few lines of code.
> - Automatically utilize state-of-the-art techniques (where appropriate) without expert knowledge.
> - Leverage automatic hyperparameter tuning, model selection/ensembling, architecture search, and data processing.
> - Easily improve/tune your bespoke models and data pipelines, or customize AutoGluon for your use-case.

In this repository we are going to see an example of how to take a sample input, pre-process it to convert it to the right format, train different algorithms and evaluate their performance, and perform inference on a validation sample. The input dataset won't be made available but, as a guideline, it follows the structure below:

| Case Details | Category | Subcategory | Sector | Tag    |
|--------------|----------|-------------|--------|--------|
| Sentence 1   | A        | AA          | S01    | Double |
| Sentence 2   | B        | BA          | S01    | Mono   |
| ...          | ...      | ...         | ...    | ...    |
| Sentence N   | Z        | ZZ          | S99    | Mono   |

## Getting started
- [SageMaker StudioLab Explainer Video](https://www.youtube.com/watch?v=FUEIwAsrMP4)
- [CodeFormer](https://auto.gluon.ai/stable/index.html)
- [New built-in Amazon SageMaker algorithms for tabular data modeling: LightGBM, CatBoost, AutoGluon-Tabular, and TabTransformer](https://aws.amazon.com/blogs/machine-learning/new-built-in-amazon-sagemaker-algorithms-for-tabular-data-modeling-lightgbm-catboost-autogluon-tabular-and-tabtransformer/)
- [Run text classification with Amazon SageMaker JumpStart using TensorFlow Hub and Hugging Face models](https://aws.amazon.com/blogs/machine-learning/run-text-classification-with-amazon-sagemaker-jumpstart-using-tensorflow-hub-and-huggingface-models/)

## Step by step tutorial

### Setup your environment

First, you need to get a [SageMaker Studio Lab](https://studiolab.sagemaker.aws/) account. This is completely free and you don't need an AWS account. Because this new service is still in Preview and AWS is looking to reduce fraud (e.g., crypto mining), you will need to wait 1-3 days for your account to be approved. You can see [this video](https://www.youtube.com/watch?v=FUEIwAsrMP4&ab_channel=machinelearnear) for more information. [Google Colab](https://colab.research.google.com/) also provides free GPU compute (NVIDIA T4/K80).

[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/machinelearnear/towards_robust_blind_face_restoration/blob/main/step_by_step.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://studiolab.sagemaker.aws/import/github/machinelearnear/towards_robust_blind_face_restoration/blob/main/step_by_step.ipynb)

Now that you have your Studio Lab account, you can follow the steps shown in `blind_face_restoration.ipynb` > [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/machinelearnear/towards_robust_blind_face_restoration/blob/main/blind_face_restoration.ipynb)

Click on `Copy to project` in the top right corner. This will open the Studio Lab web interface and ask you whether you want to clone the entire repo or just the Notebook. Clone the entire repo and click `Yes` when asked about building the `Conda` environment automatically. You will now be running on top of a `Python` environment with the basic dependencies installed.

## References

```
@article{zhou2022codeformer,
    author = {Zhou, Shangchen and Chan, Kelvin C.K. and Li, Chongyi and Loy, Chen Change},
    title = {Towards Robust Blind Face Restoration with Codebook Lookup TransFormer},
    journal = {arXiv preprint arXiv:2206.11253},
    year = {2022}
}
```

## Disclaimer
- The content provided in this repository is for demonstration purposes and not meant for production. You should use your own discretion when using the content.
- The ideas and opinions outlined in these examples are my own and do not represent the opinions of AWS.