# GAN_studies
this is a 21 week plan to study GAN

>paper reviews and final sorted report on GAN in /doc directory  
>reproduced models (in python) are in /src directory

### weekly schedule:
<details><summary>finished progress</summary>

* week 1
  scheduled: Feb. 1- 7        

    1. read first few GAN papers
        - Generative Adversarial Nets https://arxiv.org/abs/1406.2661  
NIPS 2016 tutorial: Generative Adversarial Networks https://arxiv.org/pdf/1701.00160.pdf  
Generative Adversarial Networks: An Overview https://arxiv.org/pdf/1710.07035.pdf  

finished: Feb. 19 

---

</details>

* week 2  
scheduled: Feb. 8 - 14  

    1. read paper on DCGAN: Unsupervised Representation Learning with Deep Convolutional Generative   Adversarial Networks  https://arxiv.org/pdf/1511.06434.pdf  
    2. do DCGAN on cifar10 following Keep Calm and train a GAN  
https://medium.com/@utk.is.here/keep-calm-and-train-a-gan-pitfalls-and-tips-on-training-generative-adversarial-networks-edd529764aa9  

* week 3  
scheduled: Feb. 15 - 21  

    1. read InfoGAN: https://arxiv.org/pdf/1606.03657.pdf  

    2. reproduce InfoGAN on MNIST and CelebA  
following https://github.com/taeoh-kim/Pytorch_InfoGAN  
other resources:  
https://github.com/znxlwm/pytorch-generative-model-collections  
https://github.com/eriklindernoren/PyTorch-GAN#infogan  
https://github.com/pianomania/infoGAN-pytorch  

* week 4  
scheduled: Feb. 22 - 28  

1. read Wassertein GAN: https://arxiv.org/abs/1701.07875  
and Improved training of Wassertein GANs https://arxiv.org/abs/1704.00028  
reproduce WGAN on lsun and cifar10  
following: https://github.com/martinarjovsky/WassersteinGAN  
other resources https://github.com/cedrickchee/wasserstein-gan  

2. reproduce Improved WGAN on lsun and cifar10  
following: https://github.com/jalola/improved-wgan-pytorch  
other resources https://github.com/caogang/wgan-gp  

\<basic GAN readings and experiments\>  

* week 5  
scheduled: Mar. 1 - 7  

    1. read Conditional GAN  https://arxiv.org/pdf/1411.1784.pdf  
    2. reproduce Conditional GAN   
following: https://github.com/znxlwm/pytorch-MNIST-CelebA-cGAN-cDCGAN  
other resources  
https://github.com/malzantot/Pytorch-conditional-GANs  

* week 6  
scheduled: Mar. 8 - 14  

    1. read Conditional Image Synthesis With Auxiliary Classifier GANs  https://arxiv.org/pdf/1610.09585.pdf  
    2. reproduce ACGAN  
following: https://github.com/gitlimlab/ACGAN-PyTorch 
other resources  
https://github.com/kimhc6028/acgan-pytorch  
https://github.com/TuXiaokang/ACGAN.PyTorch  
https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/acgan  
https://github.com/znxlwm/pytorch-generative-model-collections/blob/master/ACGAN.py  

* week 7  
scheduled: Mar. 15 - 21  

    1. read Stacked Generative Adversarial Networks  
https://arxiv.org/pdf/1612.04357.pdf, and  
StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks  
https://arxiv.org/pdf/1612.03242.pdf, and  
StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks  
https://arxiv.org/abs/1710.10916  
read Progressive Growing of GANs for Improved Quality, Stability, and Variation  
https://arxiv.org/abs/1710.10196  

    2. reproduce stackGAN++   
following: https://github.com/hanzhanggit/StackGAN-v2 
reproduce PGGAN   
following: https://github.com/nashory/pggan-pytorch  
other resources  
https://medium.com/@animeshsk3/msg-gan-multi-scale-gradients-gan-ee2170f55d50  
https://github.com/hanzhanggit/StackGAN-Pytorch  
https://github.com/hanzhanggit/StackGAN  
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/sgan/sgan.py  

* week 8  
scheduled: Mar. 22 - 28  

    1. read Self-Attention Generative Adversarial Networks  
https://arxiv.org/abs/1805.08318  

    2. reproduce SAGAN  
following: https://github.com/heykeetae/Self-Attention-GAN  
other resources   
https://towardsdatascience.com/not-just-another-gan-paper-sagan-96e649f01a6b  
https://medium.com/@jonathan_hui/gan-self-attention-generative-adversarial-networks-sagan-923fccde790c  

https://github.com/christiancosgrove/pytorch-sagan  
https://github.com/rosinality/sagan-pytorch  
https://github.com/ankitAMD/Self-Attention-GAN-master_pytorch  
https://github.com/christiancosgrove/pytorch-sagan/blob/master/model_mnist.py  
\<medium GAN readings and experiments (cGAN, ACGAN, stackGAN, SAGAN)\>  

  
* week 9  
scheduled: Mar. 29 - Apr. 4 (there is holiday)  
    1. read:  
Multi-Generator Generative Adversarial Nets https://arxiv.org/abs/1708.02556  
Multi-Agent Diverse Generative Adversarial Networks https://arxiv.org/abs/1704.02906  
    2. rewrite to pytorch and reproduce MGAN  
following: https://github.com/qhoangdl/MGAN (tensorflow)  
(MAD GAN in tensorflow https://github.com/rishabh135/MAD-GAN-MLCAMP/blob/master/madgan_compete_modif.py   )  
(tensorflow gan example https://github.com/tensorflow/models/tree/master/research/gan) 
* week 10  
scheduled: Apr. 5 - 11  
    1. read:  
Generative Multi-Adversarial Networks https://arxiv.org/abs/1611.01673  
MD-GAN: Multi-Discriminator Generative Adversarial Networks for Distributed Datasets  
https://arxiv.org/abs/1811.03850  

    2. rewrite to pytorch and reproduce GMAN  
following: https://github.com/iDurugkar/GMAN (tensorflow)  

* week 11  
scheduled: Apr. 12 - 18 (Midterm)  
* week 12  
scheduled: Apr. 19 - 25  
    1. 
(best results with imagenet ((possibly with PGGAN)) ) 
create GAN with multiple generator and multiple discriminator  

* week 13  
scheduled: Apr. 26 - May 2  

    1. create GAN with multiple generator and multiple discriminator  
and connect them in different ways  
aim: smaller GAN (than PGGAN?) that can create good image (~ imagenet)  

\<follow multiple GAN examples and try different approaches>  

  
* week 14  
scheduled: May 3 - 9   

    1. read Memorization Precedes Generation: Learning Unsupervised GANs with Memory Networks  
https://arxiv.org/abs/1803.01500  
Memory Replay GANs: learning to generate images from new categories without forgetting  
https://arxiv.org/abs/1809.02058  

    2. reproduce  
https://github.com/whyjay/memoryGAN  
   
「  

Goal:  
succeed in creating high quality image (imagenet pictures) with single GPU, less training hours  
try to succeed with multiple classes.  

* week 15  
scheduled: May 10 - 16  
* week 16  
scheduled: May 17 - 23  
* week 17  
scheduled: May 24 - 30  
* week 18  
scheduled: May 31 - Jun. 6  

」  

\<test new GAN ideas and search for related papers: memory; try new ideas!>  
* week 19  
scheduled: Jun. 7 - 13  
finish GAN results, write report  
* week 20  
scheduled: Jun. 14 - 20 (final) 
* week 21  
scheduled: Jun. 21 - 28  

finish GAN results, write report  

if there is time  
try  
https://medium.com/syncedreview/ai-brush-new-gan-tool-paints-worlds-2544e4e29c11  

<wrap up for conclution>  
