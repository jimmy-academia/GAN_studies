一、GAN 每週進度  

2/1- 7		  

read first few GAN papers Generative Adversarial Nets https://arxiv.org/abs/1406.2661  
NIPS 2016 tutorial: Generative Adversarial Networks https://arxiv.org/pdf/1701.00160.pdf  
Generative Adversarial Networks: An Overview https://arxiv.org/pdf/1710.07035.pdf  

2/8 - 14  

read paper on DCGAN: Unsupervised Representation Learning with Deep Convolutional Generative   Adversarial Networks  
https://arxiv.org/pdf/1511.06434.pdf  

do DCGAN on cifar10 following Keep Calm and train a GAN  
https://medium.com/@utk.is.here/keep-calm-and-train-a-gan-pitfalls-and-tips-on-training-generative-adversarial-networks-edd529764aa9  

2/15 - 21  

read InfoGAN: https://arxiv.org/pdf/1606.03657.pdf  

reproduce InfoGAN on MNIST and CelebA  
following https://github.com/taeoh-kim/Pytorch_InfoGAN  
other resources:  
https://github.com/znxlwm/pytorch-generative-model-collections  
https://github.com/eriklindernoren/PyTorch-GAN#infogan  
https://github.com/pianomania/infoGAN-pytorch  

2/22 - 28  

read Wassertein GAN: https://arxiv.org/abs/1701.07875  
and Improved training of Wassertein GANs https://arxiv.org/abs/1704.00028  

reproduce WGAN on lsun and cifar10  
following: https://github.com/martinarjovsky/WassersteinGAN  

other resources https://github.com/cedrickchee/wasserstein-gan  

reproduce Improved WGAN on lsun and cifar10  
following: https://github.com/jalola/improved-wgan-pytorch  

other resources https://github.com/caogang/wgan-gp  

<basic GAN readings and experiments>  

3/1 - 7  

read Conditional GAN  
https://arxiv.org/pdf/1411.1784.pdf  

reproduce Conditional GAN   
following: https://github.com/znxlwm/pytorch-MNIST-CelebA-cGAN-cDCGAN  

other resources  
https://github.com/malzantot/Pytorch-conditional-GANs  

3/8 - 14  

read Conditional Image Synthesis With Auxiliary Classifier GANs  
https://arxiv.org/pdf/1610.09585.pdf  

reproduce ACGAN  
following: https://github.com/gitlimlab/ACGAN-PyTorch  

other resources  
https://github.com/kimhc6028/acgan-pytorch  
https://github.com/TuXiaokang/ACGAN.PyTorch  
https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/acgan  
https://github.com/znxlwm/pytorch-generative-model-collections/blob/master/ACGAN.py  

3/15 - 21  

read Stacked Generative Adversarial Networks  
https://arxiv.org/pdf/1612.04357.pdf, and  
StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks  
https://arxiv.org/pdf/1612.03242.pdf, and  
StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks  
https://arxiv.org/abs/1710.10916  
read Progressive Growing of GANs for Improved Quality, Stability, and Variation  
https://arxiv.org/abs/1710.10196  

reproduce stackGAN++   
following: https://github.com/hanzhanggit/StackGAN-v2  

reproduce PGGAN   
following: https://github.com/nashory/pggan-pytorch  

other resources  
https://medium.com/@animeshsk3/msg-gan-multi-scale-gradients-gan-ee2170f55d50  

https://github.com/hanzhanggit/StackGAN-Pytorch  
https://github.com/hanzhanggit/StackGAN  
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/sgan/sgan.py  

3/22 - 28  

read Self-Attention Generative Adversarial Networks  
https://arxiv.org/abs/1805.08318  

reproduce SAGAN  
following: https://github.com/heykeetae/Self-Attention-GAN  

  
other resources   
https://towardsdatascience.com/not-just-another-gan-paper-sagan-96e649f01a6b  
https://medium.com/@jonathan_hui/gan-self-attention-generative-adversarial-networks-sagan-923fccde790c  

https://github.com/christiancosgrove/pytorch-sagan  
https://github.com/rosinality/sagan-pytorch  
https://github.com/ankitAMD/Self-Attention-GAN-master_pytorch  
https://github.com/christiancosgrove/pytorch-sagan/blob/master/model_mnist.py  

<medium GAN readings and experiments (cGAN, ACGAN, stackGAN, SAGAN)>  

  
3/29 - 4/4 * 有假  

read:  
Multi-Generator Generative Adversarial Nets https://arxiv.org/abs/1708.02556  
Multi-Agent Diverse Generative Adversarial Networks https://arxiv.org/abs/1704.02906  

rewrite to pytorch and reproduce MGAN  
following: https://github.com/qhoangdl/MGAN (tensorflow)  

(MAD GAN in tensorflow https://github.com/rishabh135/MAD-GAN-MLCAMP/blob/master/madgan_compete_modif.py   )  

(tensorflow gan example https://github.com/tensorflow/models/tree/master/research/gan)  

  
4/5 - 11  

read:  
Generative Multi-Adversarial Networks https://arxiv.org/abs/1611.01673  
MD-GAN: Multi-Discriminator Generative Adversarial Networks for Distributed Datasets  
https://arxiv.org/abs/1811.03850  

rewrite to pytorch and reproduce GMAN  
following: https://github.com/iDurugkar/GMAN (tensorflow)  

4/12 - 18 * 期中考週  

4/19 - 25  

(best results with imagenet ((possibly with PGGAN)) )  

create GAN with multiple generator and multiple discriminator  

4/26 - 5/2  

create GAN with multiple generator and multiple discriminator  
and connect them in different ways  
aim: smaller GAN (than PGGAN?) that can create good image (~ imagenet)  

<follow multiple GAN examples and try different approaches>  

  
5/3 - 9   

read Memorization Precedes Generation: Learning Unsupervised GANs with Memory Networks  
https://arxiv.org/abs/1803.01500  
Memory Replay GANs: learning to generate images from new categories without forgetting  
https://arxiv.org/abs/1809.02058  

reproduce  
https://github.com/whyjay/memoryGAN  
   
「  

Goal:  
succeed in creating high quality image (imagenet pictures) with single GPU, less training hours  
try to succeed with multiple classes.  

5/10 - 16  

5/17 - 23  

5/24 - 30  

5/31 - 6/6  

」  

<test new GAN ideas and search for related papers: memory; try new ideas!>  

6/7 - 13  

finish GAN results, write report  

6/14 - 20 * 期末考週  

6/21 - 28  

finish GAN results, write report  

if there is time  
try  
https://medium.com/syncedreview/ai-brush-new-gan-tool-paints-worlds-2544e4e29c11  

<wrap up for conclution>  
