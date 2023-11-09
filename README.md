## Update 11/09/23

This repo currently contains code to replicate the experiments from our original preprint ([arXiv v1](https://arxiv.org/abs/2212.09462v1), 12/19/22). An expanded version ([arXiv v2](https://arxiv.org/abs/2212.09462)) will be presented at NeurIPS 2023. Over the next few weeks, we will update the repo with code to replicate the experiments in our NeurIPS 2023 paper.

# Latent Diffusion for Language Generation

This is the official code release for

[**Latent Diffusion for Language Generation**](https://arxiv.org/abs/2212.09462).

by Justin Lovelace, Varsha Kishore, Chao Wan, Eliot Shekhtman, and Kilian Q. Weinberger

![Figure](figures/method.png)

### Abstract
Diffusion models have achieved great success in modeling continuous data modalities such as images, audio, and video, but have seen limited use in discrete domains such as language. Recent attempts to adapt diffusion to language have presented diffusion as an alternative to autoregressive language generation. We instead view diffusion as a complementary method that can augment the generative capabilities of existing pre-trained language models. We demonstrate that continuous diffusion models can be learned in the latent space of a pre-trained encoder-decoder model, enabling us to sample continuous latent representations that can be decoded into natural language with the pre-trained decoder. We show that our latent diffusion models are more effective at sampling novel text from data distributions than a strong autoregressive baseline and also enable controllable generation.

### Citation
```
@article{lovelace2022latent,
  title={Latent Diffusion for Language Generation},
  author={Lovelace, Justin and Kishore, Varsha and Wan, Chao and Shekhtman, Eliot and Weinberger, Kilian},
  journal={arXiv preprint arXiv:2212.09462},
  year={2022}
}
```


## Environment
A suitable environment can be created with the following commands. 
```bash
conda env create -f environment.yml
python -m spacy download en_core_web_sm
```

## Datasets

The dataset files for the E2E and ROCStories datasets are included in the `datasets/` directory and do not require any additional processing. The SST and AG News datasets are loaded from the HuggingFace Hub.

## Training

We provide scripts to train the diffusion models for each dataset with our default hyperparameters. Train a model with the command 
```bash
./scripts/diffusion/text_diffusion_{dataset}.sh
``` 
where dataset is one of `{roc, e2e, sst2, ag_news}`.

## Evaluation
To evaluate a trained model on the validation set, see the `scripts/diffusion/eval_text_diffusion.sh` script for an example. The `--resume_dir` argument should be updated with the path of a trained model. 


Different sampling configurations can be explored by changing the `{num_samples, sampling_timesteps, ddim_sampling_eta}` arguments. We utilize 1,000 random samples for computing the metrics in our work. Note that MAUVE scores computed with different numbers of samples are not directly comparable (see [here](https://github.com/krishnap25/mauve) for more information about MAUVE scores).

To evaluate a trained model on the test set with 5 random seeds, see the `scripts/diffusion/test_eval_text_diffusion.sh` script for an example. The only difference is that the `eval_test` flag is used instead of the `eval` flag. The `--resume_dir` argument will need to be updated as before.

## Contact
Please open an issue if you have any questions about using this repo. I will be updating the repo with the code for the classification experiment and the autoregressive baseline after the holiday season.


## Acknowledgement
This work built upon excellent open-source implementations from [Lucidrains](https://github.com/lucidrains). Specifically, we adapted his Pytorch DDPM implementation ([link](https://github.com/lucidrains/denoising-diffusion-pytorch)) and built upon his transformer implementation ([link](https://github.com/lucidrains/x-transformers)).