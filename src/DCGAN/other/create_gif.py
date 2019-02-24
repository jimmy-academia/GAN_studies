### I'm just gonna do it dirty

import argparse
import imageio

parser = argparse.ArgumentParser(description='create gifs!')
parser.add_argument('--dir', type=str, help='the directory that has only the items wanted')
parser.add_argument('--out', type=str, default='other', help='output directory')
parser.add_argument('--name', type=str)
parser.add_argument('--epochs', type=int, default=20)
args = parser.parse_args()

# to plant within code
args.epochs=100
args.dir =  'results/BASIC_MNIST/loss_plots'
args.name = 'loss_plots'
# args.dir =  'results/BASIC_MNIST/generated_imgs'
# args.name = 'generated_imgs'

if args.dir is None:
	print('WARANING: SPECIFY INPUT DIRECTORY --dir')
if args.dir is None:
	print('WARANING: SPECIFY GIF NAME --name')

## hardcode Epoch_1.png
plots = []
total = 100
for i in range(total):
	file = args.dir+'/Epoch_%s.png'%i
	plots.append(imageio.imread(file))

	if i==total-1:
		plots.append(imageio.imread(file))
		plots.append(imageio.imread(file))


filename = args.out+'/'+args.name+'.gif'
imageio.mimsave(filename, plots, fps=3)
print('%s done'%filename)