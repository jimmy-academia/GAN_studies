
# week2 paper reviews for GAN
scheduled date: Feb. 8 - Feb. 14    
<p align="right">  
<a href="README.md">back to table</a>; <a href="../README.md/#weekly-schedule">back to schedule</a>
</p>

## Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
(using new template for paper review starting here)
* **Authors:** Alec Radford, Luke Metz, Soumith Chintala  
* [**paper link**](https://arxiv.org/abs/1511.06434)  
* **publish date:** Nov 2015  

#### summary
> this work finds the best way to utilize Convolution neural network in GAN. The resulting model is one without pooling layer, uses batchnorm, relu/leaky-relu activation function.
##### chapter-wise note
(3. Approach and model architecture)
* guidelines:
	* Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
	* Use batchnorm in both the generator and the discriminator, except last of G and last and first of D.
	* Remove fully connected hidden layers
	* Use ReLU activation in generator for all layers except for the output, which uses Tanh. Use LeakyReLU activation in the discriminator for all layers.

(6. INVESTIGATING AND VISUALIZING THE INTERNALS OF THE NETWORKS)
* walking in latent space shows smooth transformation
* latent space shows vector arithmetic, e.g. can do (smiling women) - (neutral women) + (neutral men) => (smiling men), or interpolating between turn vector will turn a face from left to right.


#### branching point
* use DCGAN as feature extractor not well understood
