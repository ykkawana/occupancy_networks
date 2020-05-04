import os
import urllib
import torch
from torch.utils import model_zoo
import wandb


class CheckpointIO(object):
    ''' CheckpointIO class.

    It handles saving and loading checkpoints.

    Args:
        checkpoint_dir (str): path where checkpoints are saved
    '''
    def __init__(self, checkpoint_dir='./chkpts', **kwargs):
        self.module_dict = kwargs
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def register_modules(self, **kwargs):
        ''' Registers modules in current module dictionary.
        '''
        self.module_dict.update(kwargs)

    def save(self, filename, **kwargs):
        ''' Saves the current module dictionary.

        Args:
            filename (str): name of output file
        '''
        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        outdict = kwargs
        for k, v in self.module_dict.items():
            v = v if not hasattr(v, 'module') else v.module
            outdict[k] = v.state_dict()
        torch.save(outdict, filename)

        wandb.save(filename)

    def load(self, filename, device=None):
        '''Loads a module dictionary from local file or url.
        
        Args:
            filename (str): name of saved module dictionary
        '''
        if is_url(filename):
            return self.load_url(filename, device=device)
        else:
            return self.load_file(filename, device=device)

    def load_file(self, filename, device=None):
        '''Loads a module dictionary from file.
        
        Args:
            filename (str): name of saved module dictionary
        '''

        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        if os.path.exists(filename):
            print(filename)
            print('=> Loading checkpoint from local file...')
            state_dict = torch.load(filename)
            if 'model' not in state_dict:
                print('Detect weight file trained outside of occ env.')
                state_dict = {'model': state_dict}
            scalars = self.parse_state_dict(state_dict, device=device)
            return scalars
        else:
            raise FileExistsError

    def load_url(self, url, device=None):
        '''Load a module dictionary from url.
        
        Args:
            url (str): url to saved model
        '''
        print(url)
        print('=> Loading checkpoint from url...')
        state_dict = model_zoo.load_url(url, progress=True)
        scalars = self.parse_state_dict(state_dict, device=device)
        return scalars

    def parse_state_dict(self, state_dict, device=None):
        '''Parse state_dict of model and return scalars.
        
        Args:
            state_dict (dict): State dict of model
    '''
        """
        for k, v in self.module_dict.items():
            if k in state_dict:
                v.load_state_dict(state_dict[k])
            else:
                print('Warning: Could not find %s in checkpoint!' % k)
        """
        for k, v in self.module_dict.items():
            if k in state_dict:
                print('load parameter')
                #if False:
                if k == 'model':
                    pretrained_dict = state_dict[k]
                    model_dict = v.state_dict()
                    new_pretrained_dict = {
                        key: val
                        for key, val in pretrained_dict.items()
                        if key in model_dict
                        and model_dict[key].shape == pretrained_dict[key].shape
                    }
                    diff = set(pretrained_dict.keys()) - set(
                        new_pretrained_dict.keys())
                    print('ignored parameters')
                    for key in diff:
                        print(key, pretrained_dict[key].shape)
                    pretrained_dict_new_param = {
                        key: val
                        for key, val in model_dict.items()
                        if key not in new_pretrained_dict
                    }
                    print('new parameters')
                    for key in pretrained_dict_new_param:
                        print(key, pretrained_dict_new_param[key].shape)
                    new_pretrained_dict.update(pretrained_dict_new_param)
                    if device is not None:
                        v.load_state_dict(new_pretrained_dict)
                    else:
                        v.load_state_dict(new_pretrained_dict)

                else:
                    v.load_state_dict(state_dict[k])
        scalars = {
            k: v
            for k, v in state_dict.items() if k not in self.module_dict
        }
        print('load done')
        return scalars


def is_url(url):
    scheme = urllib.parse.urlparse(url).scheme
    return scheme in ('http', 'https')
