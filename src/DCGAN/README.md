# week2 project: Deep Convolution Generative Adversarioal Networks (DCGAN)

[Goal:] to implement the paper [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434)

## usage:
run experiments with `python experiments/<python script>`


model structure:

<img src="other/DCGAN_model.png" height=300/>

### Results
* For learning rate = 0.0002 (Adam optimizer), batch size = 128, # of epochs = 20:
<table align='center'>
<tr align='center'>
<td> GAN losses</td>
<td> Generated images</td>
</tr>
<tr>
<td><img src = 'other/loss_plots.gif'>
<td><img src = 'other/generated_imgs.gif'>
</tr>
</table>


## Tabel of Contents
* [Experiments and Results](report.md)  
* [ErrorLog](errorlog.md)

## references: 
https://github.com/togheppi/DCGAN/blob/master/README.md





other


After every 100 training iterations, the files `real_samples.png` and `fake_samples.png` are written to disk
with the samples from the generative model.

After every epoch, models are saved to: `netG_epoch_%d.pth` and `netD_epoch_%d.pth`

## Downloading the dataset
You can download the LSUN dataset by cloning [this repo](https://github.com/fyu/lsun) and running
```
python download.py -c bedroom
```

## Usage
```
usage: main.py [-h] --dataset DATASET --dataroot DATAROOT [--workers WORKERS]
               [--batchSize BATCHSIZE] [--imageSize IMAGESIZE] [--nz NZ]
               [--ngf NGF] [--ndf NDF] [--niter NITER] [--lr LR]
               [--beta1 BETA1] [--cuda] [--ngpu NGPU] [--netG NETG]
               [--netD NETD]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     cifar10 | lsun | imagenet | folder | lfw
  --dataroot DATAROOT   path to dataset
  --workers WORKERS     number of data loading workers
  --batchSize BATCHSIZE
                        input batch size
  --imageSize IMAGESIZE
                        the height / width of the input image to network
  --nz NZ               size of the latent z vector
  --ngf NGF
  --ndf NDF
  --niter NITER         number of epochs to train for
  --lr LR               learning rate, default=0.0002
  --beta1 BETA1         beta1 for adam. default=0.5
  --cuda                enables cuda
  --ngpu NGPU           number of GPUs to use
  --netG NETG           path to netG (to continue training)