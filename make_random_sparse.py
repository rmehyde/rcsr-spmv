import argparse
import random
from operator import itemgetter

def gen_mm(m, n, nnz, min, max):
	coords = []
	vals = []
	for i in range(nnz):
		row = int(random.random()*m)
		col = int(random.random()*n)
		while([row, col] in coords):
			row = int(random.random()*m)
			col = int(random.random()*n)
		coords.append([row,col])
		vals.append(random.random()*(max-min)+min)
	coords = sorted(coords, key=itemgetter(1,0))
	lines = []
	lines.append("{} {} {}\n".format(m,n,nnz))
	for i in range(nnz):
		lines.append("{} {} {}\n".format(coords[i][0], coords[i][1], vals[i]))
	return "".join(lines)

# set up program structure
parser = argparse.ArgumentParser(description="Generate a random sparse matrix and write it to a Marix Market file")
parser.add_argument('rows', metavar='m', type=int, nargs=1, help="number of rows in the matrix")
parser.add_argument('cols', metavar='n', type=int, nargs=1, help="number of columns in the matrix")
parser.add_argument('-o', '--filename', required=True, metavar='filename', nargs=1, dest="fname", help="name of matrix market file to be output")
parser.add_argument('--min', type=float, metavar='X', nargs=1, default=[0.0], help="minimum value of nonzero elements")
parser.add_argument('--max', type=float, metavar='Y', nargs=1, default=[1.0], help="maximum value of nonzero elements")
fillamts = parser.add_mutually_exclusive_group()
fillamts.add_argument('--nnz', metavar='N', type=int, nargs=1, help="precise number of nonzero elements to place in the matrix")
fillamts.add_argument('--fillpct', metavar='Z%', nargs=1, default="10%", help="percentage fill of sparse matrix")

args = parser.parse_args()

if args.nnz != None:
	data = gen_mm(args.rows[0], args.cols[0], args.nnz[0], args.min[0], args.max[0])
else:
	if(args.fillpct[0][-1]=='%'):
		nnz = int(float(args.fillpct[0][:-1])*args.rows[0]*args.cols[0]/100)
	else:
		nnz = int(float(args.fillpct[0])*m*n)
	data = gen_mm(args.rows[0], args.cols[0], nnz, args.min[0], args.max[0])
f = open(args.fname[0], 'w', encoding='utf-8')
f.write(data)
f.close()
print("Wrote a matrix of size {} x {} to {}".format(args.rows[0], args.cols[0], args.fname[0]))
