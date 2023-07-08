import torch, os, numpy as np
from .dataset.graphs import Graph
from . import model
class Processor():
    def __init__(self, args):
        self.args = args
        self.init_environment()
        self.init_device()
        self.init_model()

    def init_environment(self):
        np.random.seed(self.args['seed'])
        torch.manual_seed(self.args['seed'])
        torch.cuda.manual_seed(self.args['seed'])
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    def init_device(self):
        if type(self.args['gpus']) is int:
            self.args['gpus'] = [self.args['gpus']]
        if len(self.args['gpus']) > 0 and torch.cuda.is_available():
            self.output_device = self.args['gpus'][0]
            self.device =  torch.device('cuda:{}'.format(self.output_device))
            torch.cuda.set_device(self.output_device)
        else:
            self.output_device = None
            self.device =  torch.device('cpu')

    def init_model(self):
        graph = Graph('ntu-xsub120')
        kwargs = {
            'data_shape': [3, 6, 300, 25, 2],
            'num_class': 8,
            'A': torch.Tensor(graph.A),
            'parts': graph.parts,
        }
        self.model = model.create(self.args['model_type'], **(self.args['model_args']), **kwargs)
        self.model = torch.nn.DataParallel(
            self.model.to(self.device), device_ids=self.args['gpus'], output_device=self.output_device
        )
        pretrained_model =self.args['pretrained_path']
        if os.path.exists(pretrained_model):
            print('exist')
            checkpoint = torch.load(pretrained_model, map_location=torch.device('cpu'))
            self.model.module.load_state_dict(checkpoint['model'])
        else :
            # print(os.path.realpath(pretrained_model))
            print('not exist,plz input a correct model path')
    def eval(self,data): 
        self.model.eval()
        with torch.no_grad():
            x = torch.unsqueeze(data,0)
            x = x.float().to(self.device)
            out, _ = self.model(x)
            r = out.max(1)[1]
            a = out[0][r.item()]
            
        torch.cuda.empty_cache()
        return round(a.item(),3),r.item()
