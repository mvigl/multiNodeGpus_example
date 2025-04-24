from myMLpackage import simpleML
import argparse
import os

parser = argparse.ArgumentParser(description='')
# --- training setup ---
parser.add_argument('--lr', type=float,  help='learning rate',default=0.0001)
parser.add_argument('--bs', type=int,  help='batch size',default=128)
parser.add_argument('--ep', type=int,  help='epochs',default=100)
parser.add_argument('--num_workers', type=int,  help='num_workers',default=8)
parser.add_argument('--backend', help='NVIDIA (raven): "nccl", AMD (viper): "gloo"',default='nccl')
parser.add_argument('--out', help='directory where trained model is saved to',default='models')
# --- comet_ml logging ---
parser.add_argument('--project_name', help='your comet_ml project_name',default='multiGPUexample')
parser.add_argument('--api_key', help='your comet_ml api_key',default='')
parser.add_argument('--ws', help='your comet_ml workspace',default='mvigl')

args = parser.parse_args()
if (not os.path.exists(args.out)): os.system(f'mkdir {args.out}')

if __name__ == "__main__":
    simpleML.train_loop(args)