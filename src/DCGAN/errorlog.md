# ErrorLog

backto [README](README.md)

> Code couldn't produce correct result, found on Github: [DCGAN for MNIST](https://github.com/togheppi/DCGAN/blob/master/MNIST_DCGAN_pytorch.py) 

Comparison adjusted but not checked(correct vs mine):
1. model size could be too small ()

    self.layer_G = [(1024,4,1,0), (512,4,2,1), (256,4,2,1), (128,4,2,1), (1,4,2,1)]
    self.layer_D = [(1024,4,2,1), (512,4,2,1), (256,4,2,1), (128,4,2,1), (1,4,1,0)]
    versus
    self.layer_G = [(64,7,1,0), (32,4,2,1), (1,4,2,1)]
    self.layer_D = [(32,4,2,1), (64,4,2,1), (1,7,1,0)]
2. fake = 0, real = 1 vs. fake=1, real=0
3. learning rate = 0.0002 vs 0.0003
4.  errD_real + errD_fake = error, error.backwards vs errD_real.backwards, errD_fake.backwards
5.  D zerograd when training G
==> Found that error is due to **model**
real problem:
1. 
2.
tested to be benign:
1. 
out of which ... is vital
Ans:

---

> naming python script as copy.py cause import errors. (import torch, import numpy) within the file would go wrong, and afterwards doing import torch, import numpy would still spit errors
Came upon messages:
```
>>> import torch
Segmentation fault (core dumped)

AttributeError: type object 'torch._C.FloatTensorBase' has no attribute 'numpy' 
```
could be related to   
https://github.com/python/cpython/blob/master/Lib/copy.py  
https://docs.python.org/3/library/copy.html  
search term [shallow and deep copy]

Ans: rename the script and problem is cleared

> sf