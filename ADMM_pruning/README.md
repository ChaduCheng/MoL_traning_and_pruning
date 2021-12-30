
Prerequisites
-----------------------

code is compatible with pytorch 1.0.0



Train a model in natural setting/adversarial setting
-----------------------


main.py or adv_main.py for main program, natural setting and adversarial setting respectively

eval.py for quick checking of the sparsity and do some other stuff

config.yaml.example template of the configuration file. One for each dataset.

run.sh.example  template script for running the code.


Compression in adversarial setting are only supported for CIFAR10 and Tiny-ImageNet 

Run the code in ADMM_examples to achieve the adversarial training + pruning

sh run.sh.example

