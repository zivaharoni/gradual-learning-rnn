# Gradual Learning of Deep Recurrent Neuarl Networks

This repository is a self-contained code of the work in the Gradual Learning of Deep Recurrent Neural Networks.
The package contains a python implementation of the model as depicted in (https://arxiv.org/abs/1708.08863) with the present additions and changes:

- Layerwise gradient clipping was added, clipping gradients of each network element (embeddings, layer_0, layer_1, ...) by setting flag '--LWGC=True'.
- The LWGC coefficient are set by '--lwgc_grad_norm=[<emb_max_grad_norm>, <layer0_max_grad_norm>, ...]'
- Averaged stochastic gradient descent in now supported by setting flag '--opt=asgd'
- AR (activation regularization) is nor supported with coefficient set by '--AR=<coef>'
- TAR (temporal activation regularization) is nor supported with coefficient set by '--TAR=<coef>'
- Pretrained model with one missing layer can be inherited using setting 'restore=True' and 'ckpt_file=<file_name>' flags.


### Penn TreeBank
The file ptb_config.py contains all the configurations that were tested in the article. Choosing one of them is done by the model flag from the command line.

The possible models names are:

+ LWGC - two-layered LSTM trained gradually with layerwise gradient clipping.
+ GL - two-layered LSTM trained gradually.
+ LAD - two-layered LSTM trained with Layer-wise Adjusted Dropout.
+ GL_LAD - two-layered LSTM trained gradually with Layer-wise Adjusted Dropout.
+ Deep_GL_LAD - five-layered LSTM trained gradually with Layer-wise Adjusted Dropout.

To evaluate one of the configurations run the following:

`python ptb_model.py --model <model_name>`