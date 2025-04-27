import os, pprint
import argparse
from argparse import ArgumentParser
import json
import sys
import csv

from src.vad_trainer import SingleDatasetTrainer

import logging
logging.basicConfig(level=logging.ERROR)

pp = pprint.PrettyPrinter(indent=1, width=90)

def main():
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    arg_ = parser.parse_args()
    if arg_.config == None:
        raise NameError("Include a config file in the argument please.")

    #Getting configurations
    with open(arg_.config) as config_file:
        args = json.load(config_file)
    args = argparse.Namespace(**args)

    sdt = SingleDatasetTrainer(args)
    sdt.train()


if __name__ == "__main__":
    main()