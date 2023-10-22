# ConPrompt
Official implementation of the paper "ConPrompt: Pre-training a Language Model with Machine-Generated Data for Implicit Hate Speech Detection"

## Preprocess ToxiGen Dataset
We load the [ToxiGen](https://huggingface.co/datasets/skg/toxigen-data) dataset and preprocess the dataset for pre-training using preprocess_toxigen.ipynb.

The preprocess includes: 
- load train subset of ToxiGen dataset (250,951 training examples)
- remove some invalid values (17 examples, resulting in 250,934 training examples)
- anonymize private information such as email address, urls, and user or channel mention
    - We follow the implementation in https://github.com/dhfbk/hate-speech-artifacts
- set a positive sample for each machine-generated statements
- save a csv file (data/conprompt_pre-train_dataset.csv) for pre-training

## Pre-train a BERT using ConPrompt
You can pre-train a BERT using the proposed pre-training approach (**ConPrompt**) on ToxiGen dataset by:
```
bash run_conprompt.sh
```
run_conprompt.sh is an example code that we used to train our pre-trained model (i.e., **ToxiGen-ConPrompt**).
After pre-training is done using our example code, a model will be saved at `result/ToxiGen-ConPrompt`.
You can convert the saved checkpoint to Huggingface's checkpoint format by:
```
python simcse_to_huggingface.py --path result/ToxiGen-ConPrompt
```
We will release the proposed pre-trained model (ToxiGen-ConPrompt) in [Huggingface's Model Hub](https://huggingface.co/models) shortly.

## Fine-tune and evaluate ToxiGen-ConPrompt on implicit hate speech datasets
We fine-tune ToxiGen-ConPrompt on an implicit hate speech dataset and evaluate it on other implicit hate speech datasets (i.e., cross-dataset evaluation) to validate its generalization ability.

You can refer the code in https://github.com/youngwook06/ImpCon to fine-tune ToxiGen-ConPrompt and evaluate it on implicit hate speech datasets.

## Ethical Considerations
### Privacy Issue
Before pre-training, we found out that some private information such as URLs exists in the machine-generated statements in ToxiGen.
We anonymize such private information before pre-training to prevent any harm to our society.
You can refer to the anonymization code we used in preprocess_toxigen.ipynb and we strongly emphasize to anonymize private information before using machine-generated data for pre-training.

### Potential Misuse
The pre-training source of ToxiGen-ConPrompt includes toxic statements.
While we use such toxic statements on purpose to pre-train a better model for implicit hate speech detection, the pre-trained model needs careful handling.
Here, we states some behavior that can lead to potential misuse so that our model is used for the social good rather than misued unintentionally or maliciously.

- As our model was trained with the MLM objective, our model might generate toxic statements with its MLM head
- As our model learned representations regarding implicit hate speeches, our model might retrieve some similar toxic statements given a toxic statement.

While these behavior can lead to social good e.g., constructing training data for hate speech classifiers, one can potentially misuse the behaviors.

**We strongly emphasize the need for careful handling to prevent unintentional misuse and warn against malicious exploitation of such behaviors.**


## Acknowledgements
- We use the [ToxiGen](https://huggingface.co/datasets/skg/toxigen-data) dataset as a pre-training source to pre-train our model. You can refer to the paper [here](https://aclanthology.org/2022.acl-long.234/).
- We anonymize private information following the code from https://github.com/dhfbk/hate-speech-artifacts.
- Our pre-training code is based on the code from https://github.com/princeton-nlp/SimCSE with some modification.
- We use the code from https://github.com/youngwook06/ImpCon to fine-tune and evaluate our model.



