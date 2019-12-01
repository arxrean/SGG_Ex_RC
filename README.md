# SGG_Ex_RC
1. This is an unofficial implementation for CVPR19:Scene Graph Generation with External Knowledge and Image Reconstruction.
2. Part of the code is based on [FactorizableNet](https://github.com/yikang-li/FactorizableNet)
3. All modules in the paper have been done. Code can be run without error. Only support VRD dataset now.
4. There is still something wrong for training loss. If I train FactorizableNet from scratch, all object proposals will be predicted as 0(background), with or without added modules.

# Instructions
1. Download the VRD dataset as same as [FactorizableNet](https://github.com/yikang-li/FactorizableNet).
2. Download Glove from [GloVe](https://github.com/stanfordnlp/GloVe). Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download) is used.
3. CUDA_VISIBLE_DEVICES=0 python -m pdb train_FN.py --path_opt options/models/VRD.yaml --rpn output/RPN_VRD.h5
--CON_use 0 (concept network and KB)
--RC_use 0 (reconstruct images from objects)
--GAN_use 0 (G/D training loss)
