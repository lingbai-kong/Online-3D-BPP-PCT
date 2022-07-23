import sys 
sys.path.append("..")
from tools import get_args
from model_loader import nnModel

args = get_args()

nmodel = nnModel('pretrained_models/PCT_setting1.pt', args)