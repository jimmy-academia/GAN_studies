######################################################
###  Created by Jimmy Yeh, Made In Taiwan          ###
###  for the sole purpose of being excellent       ###
###  What do people usually put here? this doesn't look pretty
######################################################

# description:
#     everything other than prepare dataset and create model
#     ex: training, 
#         show training history
#         show generated result
#         evaluating???
#         (and different configuration of above by config.py)


## --> go check opc the ram faimily
### ganmain.py
## ref:
# https://github.com/pytorch/examples

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

import os 
import sys
import argparse
from random import randint

from models import GAN
from utils import progress_bar

parser = argparse.ArgumentParser(description='VAE Training')
parser.add_argument('--data', default='mnist', type=str, help='dataset')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
args = parser.parse_args()

print(args.data)
if not any(args.data == i for i in ['mnist', 'cifar10']): 
    print('WRONG DATASET KEYWORD, use "mnist" "cifar10"')
    sys.exit()
name_path_dict = {
    'cifar10': 'c10',
    'cifar100': 'c100',
    'mnist': 'mst',
}

dataset_dir = '../datasets/'
DPATH = dataset_dir + args.data
checkpoint_dir = './checkpoint/'
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
CPATH = checkpoint_dir + 'gan_'+name_path_dict[args.data]+'.t7'
RESUME = False
if os.path.isfile(CPATH):
    RESUME = True
image_dir = './img/gan/'
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)
image_dir = image_dir + name_path_dict[args.data]+'/'
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)

print('==> Preparing data..')
print('GAN Working on: ', args.data)

transform_data = transforms.Compose([
    transforms.Resize(64),
    # transforms.CenterCrop(64), # later check lsun imagenet
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

if args.data =='mnist':
    dataset = torchvision.datasets.MNIST(root=DPATH, train=True, download=True, transform=transform_data)
    img_channel_num = 1
elif args.data =='cifar10':
    dataset = torchvision.datasets.CIFAR10(root=DPATH, train=True, download=True, transform=transform_data)
    dataset = [i for i in dataset if i[1]==3]
    img_channel_num = 3
elif args.data =='cifar100':
    dataset = torchvision.datasets.CIFAR100(root=DPATH, train=True, download=True, transform=transform_data)
    img_channel_num = 3
    pass
elif args.data =='CELEBA':
    pass

#plane, car, bird, cat, deer, dog, frog, horse, ship, truck
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)


print('==> Building model..')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0

net = GAN(img_channel_num)
if device == 'cuda':
    net.cuda()
    cudnn.benchmark = True

if RESUME:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(CPATH)
    net.generator.load_state_dict(checkpoint['gen_net'])
    net.discriminator.load_state_dict(checkpoint['dis_net'])
    start_epoch = checkpoint['epoch']


real_label = 0
fake_label = 1

Gen_optimizer = optim.Adam(net.generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
Dis_optimizer = optim.Adam(net.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

def train(epoch):
    print('\nEpoch: %d'%epoch)
    for batch_idx, (inputs, __) in enumerate(dataloader):

        # Discriminator  maximize log(D(x)) + log(1 - D(G(z)))
        net.discriminator.zero_grad()

        inputs = inputs.to(device)
        batch_size = inputs.shape[0]
        label = torch.full((batch_size, ), real_label, device=device)
        out = net.discriminator(inputs)
        err_Dis_real = criterion(out, label)
        err_Dis_real.backward(retain_graph=True)
        Dis_out = out.mean().item()

        z = torch.randn(batch_size, 100, 1, 1, device=device)
        fake_inputs = net.generator(z)
        label.fill_(fake_label)
        
        out = net.discriminator(fake_inputs)
        err_Dis_fake = criterion(out, label)
        err_Dis_fake.backward(retain_graph=True)

        err_Dis = err_Dis_fake + err_Dis_real
        Dis_optimizer.step()

        # Generator  maximize log(D(G(z)))
        for _ in range(1):
            net.generator.zero_grad()
            label.fill_(real_label)
            out = net.discriminator(fake_inputs)
            Dis_gen_out = out.mean().item()

            err_Gen = criterion(out, label)
            err_Gen.backward(retain_graph=True)
            Gen_optimizer.step()

        message = 'errD:%.4f,errG:%.4f,D(x):%.4f,D(G(z)):%.4f' \
            %(err_Dis.item(), err_Gen.item(), Dis_out, Dis_gen_out)
        progress_bar(batch_idx, len(dataloader), message)

        ## save
        state = {
            'gen_net': net.generator.state_dict(),
            'dis_net': net.discriminator.state_dict(),
            'epoch' : epoch,
        }
        torch.save(state, CPATH)
    ##save
    if (epoch+1)%10 ==0:
        with torch.no_grad():
            r = randint(0, len(dataset))
            real_img = dataset[r][0][:]
            save_image(real_img.cpu(), image_dir+'real_'+str(epoch)+'.png')
            
            z = torch.randn(1, 100, 1, 1, device=device)
            fake_img = net.generator(z)
            save_image(fake_img.cpu(), image_dir+'gen_'+str(epoch)+'.png')

for epoch in range(start_epoch, start_epoch+200):
    train(epoch)



##################################






def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

fixed_z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)    # fixed noise
fixed_z_ = Variable(fixed_z_.cuda(), volatile=True)
def show_result(num_epoch, show = False, save = False, path = 'result.png', isFix=False):
    z_ = torch.randn((5*5, 100)).view(-1, 100, 1, 1)
    z_ = Variable(z_.cuda(), volatile=True)

    G.eval()
    if isFix:
        test_images = G(fixed_z_)
    else:
        test_images = G(z_)
    G.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

# training parameters
batch_size = 128
lr = 0.0002
train_epoch = 20

# data_loader
img_size = 64
transform = transforms.Compose([
        transforms.Scale(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

# network
G = generator(128)
D = discriminator(128)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G.cuda()
D.cuda()

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# results save folder
if not os.path.isdir('MNIST_DCGAN_results'):
    os.mkdir('MNIST_DCGAN_results')
if not os.path.isdir('MNIST_DCGAN_results/Random_results'):
    os.mkdir('MNIST_DCGAN_results/Random_results')
if not os.path.isdir('MNIST_DCGAN_results/Fixed_results'):
    os.mkdir('MNIST_DCGAN_results/Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []
num_iter = 0

print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    D_losses = []
    G_losses = []
    epoch_start_time = time.time()
    for x_, _ in train_loader:
        # train discriminator D
        D.zero_grad()

        mini_batch = x_.size()[0]

        y_real_ = torch.ones(mini_batch)
        y_fake_ = torch.zeros(mini_batch)

        x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
        D_result = D(x_).squeeze()
        D_real_loss = BCE_loss(D_result, y_real_)

        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        z_ = Variable(z_.cuda())
        G_result = G(z_)

        D_result = D(G_result).squeeze()
        D_fake_loss = BCE_loss(D_result, y_fake_)
        D_fake_score = D_result.data.mean()

        D_train_loss = D_real_loss + D_fake_loss

        D_train_loss.backward()
        D_optimizer.step()

        # D_losses.append(D_train_loss.data[0])
        D_losses.append(D_train_loss.data[0])

        # train generator G
        G.zero_grad()

        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        z_ = Variable(z_.cuda())

        G_result = G(z_)
        D_result = D(G_result).squeeze()
        G_train_loss = BCE_loss(D_result, y_real_)
        G_train_loss.backward()
        G_optimizer.step()

        G_losses.append(G_train_loss.data[0])

        num_iter += 1

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time


    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses))))
    p = 'MNIST_DCGAN_results/Random_results/MNIST_DCGAN_' + str(epoch + 1) + '.png'
    fixed_p = 'MNIST_DCGAN_results/Fixed_results/MNIST_DCGAN_' + str(epoch + 1) + '.png'
    show_result((epoch+1), save=True, path=p, isFix=False)
    show_result((epoch+1), save=True, path=fixed_p, isFix=True)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G.state_dict(), "MNIST_DCGAN_results/generator_param.pkl")
torch.save(D.state_dict(), "MNIST_DCGAN_results/discriminator_param.pkl")
with open('MNIST_DCGAN_results/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path='MNIST_DCGAN_results/MNIST_DCGAN_train_hist.png')

images = []
for e in range(train_epoch):
    img_name = 'MNIST_DCGAN_results/Fixed_results/MNIST_DCGAN_' + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave('MNIST_DCGAN_results/generation_animation.gif', images, fps=5)