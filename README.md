This is a work in progress.

# Requirements
This was run on a rocm (AMD) based system, therefore the requirements.txt file needs to be modified for NVIDA GPUs. Install all requirements by running `pip install -r requirements.txt`

# Datasets
NRC-VAD Lexicon can be found [here](https://emilhvitfeldt.github.io/textdata/reference/lexicon_nrc_vad.html)

SemEval can be found [here](https://www.kaggle.com/datasets/azzouza2018/semevaldatadets)

laion2B can be found [here](https://huggingface.co/datasets/laion/laion2B-en-aesthetic) and leverage the `download_laion2B-en-aesthetic.ipynb` notebook to download the images.

EmoBank can be found [here](https://www.kaggle.com/datasets/jackksoncsie/emobank)

# VAD Training
Once all datasets are downloaded into the src/../../datasets directory, enter the following into the command line to start the training `python src/main.py --config config.txt` or `PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512 python src/main.py --config config.txt`

# VAD Inference
For incremental testing of the VAD model, you can leverage the `vad-inference.ipynb` notebook

# GAN Training
Once all the datasets are downloaded into src/../../datasets directory, enter the following into the command line to start the training `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python gigagan-main.py` or you can leverage the `gigagan-training.ipynb` notebook.
