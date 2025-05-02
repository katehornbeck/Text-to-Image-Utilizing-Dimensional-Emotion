This is a work in progress.

# VAD Training
Once all datasets are downloaded into the src/../../datasets directory, enter the following into the command line to start the training `python src/main.py --config config.txt` or `PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512 python src/main.py --config config.txt`

`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python gigagan-main.py`
