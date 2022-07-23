import sys 
sys.path.append("..")
from tools import load_policy,get_args
from model import DRL_GAT
import torch
import tools

class nnModel(object):
    def __init__(self,url,args):
        self.args = get_args()
        
        if self.args.no_cuda:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda', self.args.device)
            torch.cuda.set_device(self.args.device)

        self.PCT_policy =  DRL_GAT(self.args)
        self.PCT_policy =  self.PCT_policy.to(self.device)
        
        self.PCT_policy=load_policy(url, self.PCT_policy)
        self.PCT_policy.eval()
        print('Pre-train model loaded!', url)
    def evaluate(self,obs,_):
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(dim=0)
        batchX = torch.arange(self.args.num_processes)
        all_nodes, leaf_nodes = tools.get_leaf_nodes_with_factor(obs, 
                                                                 self.args.num_processes,
                                                                 self.args.internal_node_holder,
                                                                 self.args.leaf_node_holder)
        with torch.no_grad():
            selectedlogProb, selectedIdx, policy_dist_entropy, value = self.PCT_policy(all_nodes, True, normFactor = 1)
        return value,selectedlogProb,selectedIdx[0][0],leaf_nodes[0]