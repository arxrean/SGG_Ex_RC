# Notice
1. This is an unofficial implementation for CVPR19:Scene Graph Generation with External Knowledge and Image Reconstruction.

2. Code is based on [FactorizableNet](https://github.com/yikang-li/FactorizableNet).

3. All modules in the paper have been done. Code can run without error. Only support VRD dataset now.

4. There is still something wrong with training loss for original code in FactorizableNe. If I train the model with only RPN pretrained, all proposals will be predicted as 0(background) after one iteration, with or without added modules in this repo.

# Instruction
1. Download VRD data as same as in [FactorizableNet](https://github.com/yikang-li/FactorizableNet).
2. Compile the Faster-RCNN lib as same as in [FactorizableNet](https://github.com/yikang-li/FactorizableNet).
3. Download Glove txt from [GloVe](https://github.com/stanfordnlp/GloVe) and put it into ./data/Glove. glove.840B.300d.zip is used.
4. CUDA_VISIBLE_DEVICES=0 python train_FN.py --path_opt options/models/VRD.yaml --rpn output/RPN_VRD.h5 
--CON_use 0 | 1 (Concept network for getting KB and refining object proposals.)
--RC_use 0 | 1 (Reconstruct images from object proposals)
--GAN_use 0 | 1 (G/D training)
