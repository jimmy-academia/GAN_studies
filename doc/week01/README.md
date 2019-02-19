
# week1 paper reviews for GAN
scheduled date: Feb. 1 - Feb. 7    

> paper list:  
[ Generative Adversarial Nets ](#Generative-Adversarial-Nets)  
[NIPS 2016 tutorial: Generative Adversarial Networks](#NIPS-2016-tutorial-Generative-Adversarial-Networks)  
[Generative Adversarial Networks: An Overview ](#Generative-Adversarial-Networks-An-Overview)  


## Generative Adversarial Nets     
[paper link](https://arxiv.org/abs/1406.2661)

> This is the first proposal of GAN by Ian Goodfellow et al. 
> GAN, generative adversarial network, is a new framework including a of a minimax game of two players: a generator (G) and a discriminator (D). 
> Math is provided to illustrate that p_G (distribution of sample generated) will tend towards p_data (distribution of data)

#### selected key points:
1. previous works on generative models uses Markov chain Monte Carlo methods (problemetic). GAN doesn't.
2. the value function is <p align="center"><img src="/doc/week01/tex/13aed6a8e67d831b164d183c852a7ae5.svg?invert_in_darkmode&sanitize=true" align=middle width=454.93576095pt height=18.17354385pt/></p> and the game is <p align="center"><img src="/doc/week01/tex/eee58ae08fc4112f8c8c041ef77899ed.svg?invert_in_darkmode&sanitize=true" align=middle width=123.79459784999999pt height=22.931502pt/></p>
3. Global optimun for D this game is <p align="center"><img src="/doc/week01/tex/5a47a14be8ab8056045b018a263e9556.svg?invert_in_darkmode&sanitize=true" align=middle width=186.26920785pt height=39.428498999999995pt/></p> 
4. under the condition in 3. global optimum for is acheived when <img src="/doc/week01/tex/2e67499e7826e2612a2cd14ece14eae9.svg?invert_in_darkmode&sanitize=true" align=middle width=72.17626349999998pt height=14.15524440000002pt/>
#### branching points:
1. what is Markov chain Monte Carlo methods 
2. proof of keypoint 4 is related to KL divergence and Jensen-Shannon divergence
3. there is gap between optimizing <img src="/doc/week01/tex/c792a6f8388d7a9d1305e9cbd7aabed2.svg?invert_in_darkmode&sanitize=true" align=middle width=15.09653639999999pt height=14.15524440000002pt/> and <img src="/doc/week01/tex/442b66a193e68f9acbebdc7d3d04a580.svg?invert_in_darkmode&sanitize=true" align=middle width=14.54286239999999pt height=22.831056599999986pt/>, the gap is covered by the effectiveness of neural networks.
4. data points are discrete and could have very different properties with small shifts (eg adversarial). could it be a problem for GAN to use <img src="/doc/week01/tex/a8b88154842060731b33a4a01984e7fd.svg?invert_in_darkmode&sanitize=true" align=middle width=57.34250939999999pt height=24.65753399999998pt/>, a continuous distribution in theory.

### NIPS 2016 tutorial: Generative Adversarial Networks 
[paper link](https://arxiv.org/pdf/1701.00160.pdf)
> This is a tutorial of GAN by Ian Goodfellow
> Content includes: why GAN, details on GAN vs related model, math of GAN, tips/tricks, research frontiers etc.

#### selected key points:
1. GAN (or other generative model) can handle multi-modal outputs, which if averaged won't be correct (ex predict next video frame of turning head, wrong if you do [turn left + turn right]/2)
2. DCGAN are fundamental architechture for recent GANs, key insights includes: batch normalization in both D and G (last layer of D and first of G is not batch normalized); use transposed convolution, no pooling or unpooling katers; use ADAM rather than SGD with momentum
3. Tips and Tricks:  
    a. train with labels: can help model gain more info  
    b. one-sided label smoothing: only on true side (1->0.9); (0->0.1) not good  
    c. batch normalization oscilates results when batch too small. reference batch normalization sampled at the start, or virtual batch normalization (= ref + example) solves problem  
    d. Author believes that GAN wors by estimating ratio fo data density and model density, and would only work when D is optimal. When D too accurate, gradient for G vanishes. However still let D>G (by k vs 1 step update for each) and use parameterization of the game. D is often deeper/more layer in practice.   
    e. parameterization: use $-logD(G(z))$ instead of $log(1-D(G(z)))$  
4. Research frontiers:
    a. None covergence is an issue the gap between optimizing 𝑝𝑔 (function space) and 𝜃𝑔 (parameter) blocks the theorem for GAN to gurantee convergence. There is currently no theoretical proof or argument as to whether GAN game should converge or not  
    b. One of main convergance problem is: **Mode Collapse**, aka the Helvetica scenerio. This happens when (why this no work? hello hello)   $$G^* = \min\limits_G\max\limits_DV(G,D)$$ turns to $$G^* = \max\limits_D\min\limits_GV(G,D)$$ in the later G simply always produce one of the optimal mode.  
    c. Minibatch features (check is samples in minibatch is similar to one another) works well for preventing mode collapes. It is recommended to copy the code for this. Other solution is unrolled GAN (can't scale for ImageNet), stackGAN  
    d. minibatch feature work so well for mode collapse that author suggests we work other problems including difficulties of counting, perspective, global structure (various unphsical/unbiological images generated)   
5. other frontiers
    (p) GAN research could enhance AI research with games. Connection to RL is promising  
    (p) semi-supervised learning (additional fake image class)  
    (p) using the code. Encoder for arbitrary img -> code hasn't succeed but infoGan is useful  
    (§)  plug and play not well understood yet and is skipped  
6. <img src="/doc/week01/tex/4d2791386c95386ce91fa568e0e38dcb.svg?invert_in_darkmode&sanitize=true" align=middle width=91.33938494999998pt height=24.65753399999998pt/> is a simple example of saddle point. If solve for minimax game we would get two set of sinusoids and essentialy a circular orbit that never reaches equilibrium (x=y=0) 
7. "For GANs, there is no theoretical prediction as to whether simultaneous gradient descent should converge or not. Settling this theoretical question and developting algorithms guaranteed to converge, remain important open research problems"


#### branching points:
1. is it really not ideal to smooth label for fake samples? should test!!!
2. code and best practice for reference batch normalization to be done
3. parameterization not clearly understood, think that the effect is for D to have larger gradients for G to use? is there technical term for this?
4. difficulties of counting (ex 3 eyes), perspective, global structure seems to requires common sence. How is GAN supposed to know except by luck?
5. is convergence problem settled yet?



## Generative Adversarial Networks: An Overview 
[paper link]( https://arxiv.org/pdf/1710.07035.pdf)
>This paper reviews and introduces various types of GAN.
>Topics include FC/DCGAN, conditional GANs, Adversarial Autoencoders; 
>Brief review of GAN theory, training tricks Alternative formulations; Applications of GAN, classification, image synthesis, img-to-img, super-resolution. 
>Finally it discusses some open questions on GAN. 

#### selected key points:
1.  in Conditional GAN, D output T/F w/ condition of class
in InfoGAN D output T/F and class
2. Bidirectional GAN: G, D and E (Encoder)
    D discrimate between G(data' <- noice)/ E(data -> latent code)
3. WGAN prevents vanishing gradients, and is easy to train

#### branching points:
1. What is Adversarial Autoencoder? Not clear from this paper, check https://arxiv.org/pdf/1511.05644.pdf
2. with (Bidirectional) GAN we have img->latent code, can we compare latent code with memory of class to do classification??
3. GAN need convergence to saddle point. Lee et al. (Gradient descent only converges to minimizers) showed that train GAN with gradient descent would not converge with probability one. Hope is found in second-order optimizers? Among which Newton-type methods is to slow. Arora et al. (Generalization and equilibrium in GAN) propose 'neural net distance', for it is argued that even when GAN appear to have converged, trained distribution could still be far from target.