
# week1 paper reviews for GAN
scheduled date: Feb. 1 - Feb. 7
<p align="right">  
<a href="README.md">back to table</a>; <a href="../README.md/#weekly-schedule">back to schedule</a>
</p>

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
2. the value function is $$V(D,G) = E_{x\sim p_{data}(x)}[log(D(x))]+E_{z\sim p_{z}(z)}[log(1-D(G(z)))]$$ and the game is $$\min\limits_{G}\max\limits_{D}V(D,G)$$
3. Global optimun for D this game is $$ D^*_G(x) = \dfrac{p_{data}(x)}{p_{data}(x)+p_g(x)}$$ 
4. under the condition in 3. global optimum for is acheived when $p_g = p_{data}$
#### branching points:
1. what is Markov chain Monte Carlo methods 
2. proof of keypoint 4 is related to KL divergence and Jensen-Shannon divergence
3. there is gap between optimizing $p_g$ and $\theta_g$, the gap is covered by the effectiveness of neural networks.
4. data points are discrete and could have very different properties with small shifts (eg adversarial). could it be a problem for GAN to use $p_{data}(x)$, a continuous distribution in theory.

## NIPS 2016 tutorial: Generative Adversarial Networks 
[paper link](https://arxiv.org/pdf/1701.00160.pdf)
> This is a tutorial of GAN by Ian Goodfellow
> Content includes: why GAN, details on GAN vs related model, math of GAN, tips/tricks, research frontiers etc.

#### selected key points:
1. GAN (or other generative model) can handle multi-modal outputs, which if averaged won't be correct (ex predict next video frame of turning head, wrong if you do [turn left + turn right]/2)
2. DCGAN are fundamental architechture for recent GANs, key insights includes: batch normalization in both D and G (last layer of D and first of G is not batch normalized); use transposed convolution, no pooling or unpooling katers; use ADAM rather than SGD with momentum  
3. $V(x,y) = xy$ is a simple example of saddle point. If solve for minimax game we would get two set of sinusoids and essentialy a circular orbit that never reaches equilibrium (x=y=0) 
4. "For GANs, there is no theoretical prediction as to whether simultaneous gradient descent should converge or not. Settling this theoretical question and developting algorithms guaranteed to converge, remain important open research problems"

#### useful summary:
§ Tips and Tricks: 
1. train with labels: can help model gain more info  $-logD(G(z))$
2. one-sided label smoothing: only on true side (1->0.9); (0->0.1) not good  
3. batch normalization oscilates results when batch too small. reference batch normalization sampled at the start, or virtual batch normalization (= ref + example) solves problem  
4. Author believes that GAN wors by estimating ratio fo data density and model density, and would only work when D is optimal. When D too accurate, gradient for G vanishes. However still let D>G (by k vs 1 step update for each) and use parameterization of the game. D is often deeper/more layer in practice.   
5. parameterization: use $-logD(G(z))$ instead of $log(1-D(G(z)))$    

§ Research frontiers:  
1. None covergence is an issue the gap between optimizing 𝑝𝑔 (function space) and 𝜃𝑔 (parameter) blocks the theorem for GAN to gurantee convergence. There is currently no theoretical proof or argument as to whether GAN game should converge or not  
2. One of main convergance problem is: **Mode Collapse**, aka the Helvetica scenerio. This happens when $$G^* = \min\limits_G\max\limits_DV(G,D)$$ turns to $$G^* = \max\limits_D\min\limits_GV(G,D)$$ in the later G simply always produce one of the optimal mode.  
3. Minibatch features (check is samples in minibatch is similar to one another) works well for preventing mode collapes. It is recommended to copy the code for this. Other solution is unrolled GAN (can't scale for ImageNet), stackGAN  
4. minibatch feature work so well for mode collapse that author suggests we work other problems including difficulties of counting, perspective, global structure (various unphsical/unbiological images generated)   

§ other frontiers  
1. GAN research could enhance AI research with games. Connection to RL is promising  
2. semi-supervised learning (additional fake image class)  
3. using the code. Encoder for arbitrary img -> code hasn't succeed but infoGan is useful  
4. plug and play not well understood yet and is skipped  

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