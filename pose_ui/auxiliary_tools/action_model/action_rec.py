import os, yaml
from .src.processor import Processor


def action_rec():
    # Loading parameters
    args = get_parameters(2001)  #load parameters from config, update parameters with *.yaml
    p = Processor(args)
    return p
    #p.start()

def get_parameters(file_name):

    if os.path.exists('./auxiliary_tools/action_model/configs/{}.yaml'.format(file_name)):
        with open('./auxiliary_tools/action_model/configs/{}.yaml'.format(file_name), 'r') as f:
            try:
                yaml_arg = yaml.load(f, Loader=yaml.FullLoader)
            except:
                yaml_arg = yaml.load(f)
    else:
        raise ValueError('Do NOT exist this file in \'configs\' folder: {}.yaml!'.format(file_name))
    return yaml_arg

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    action_rec()
