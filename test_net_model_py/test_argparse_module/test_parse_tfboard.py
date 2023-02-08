import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("square", help="display a square of a given number",type=int)
args = parser.parse_args()
args.experiment_path = os.path.join('./','experiments') # 可在参数解析后再额外添加一些参数arg
print(args.square**2, args.experiment_path) # square参数就是位置参数