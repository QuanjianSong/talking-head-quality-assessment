import fnmatch
import re
# from pyiqa.default_model_configs import DEFAULT_CONFIGS
from extra_feats.custom_default_model_configs import DEFAULT_CONFIGS

from pyiqa.utils import get_root_logger
from pyiqa.models.inference_model import InferenceModel
import torch

from collections import OrderedDict
# from pyiqa.default_model_configs import DEFAULT_CONFIGS
from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.utils.img_util import imread2tensor

from pyiqa.losses.loss_util import weight_reduce_loss
from pyiqa.archs.arch_util import load_pretrained_network


class CustomInferenceModel(torch.nn.Module):
    """Common interface for quality inference of images with default setting of each metric."""

    def __init__(
            self,
            metric_name,
            as_loss=False,
            loss_weight=None,
            loss_reduction='mean',
            device=None,
            seed=123,
            **kwargs  # Other metric options
    ):
        super(CustomInferenceModel, self).__init__()

        self.metric_name = metric_name

        # ============ set metric properties ===========
        self.lower_better = DEFAULT_CONFIGS[metric_name].get('lower_better', False)
        self.metric_mode = DEFAULT_CONFIGS[metric_name].get('metric_mode', None)
        if self.metric_mode is None:
            self.metric_mode = kwargs.pop('metric_mode')
        elif 'metric_mode' in kwargs:
            kwargs.pop('metric_mode')
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.as_loss = as_loss
        self.loss_weight = loss_weight
        self.loss_reduction = loss_reduction

        # =========== define metric model ===============
        net_opts = OrderedDict()
        # load default setting first
        if metric_name in DEFAULT_CONFIGS.keys():
            default_opt = DEFAULT_CONFIGS[metric_name]['metric_opts']
            net_opts.update(default_opt)
        # then update with custom setting
        net_opts.update(kwargs)
        network_type = net_opts.pop('type')
        self.net = ARCH_REGISTRY.get(network_type)(**net_opts)
        self.net = self.net.to(self.device)
        self.net.eval()

        self.seed = seed

        self.dummy_param = torch.nn.Parameter(torch.empty(0)).to(self.device)
    
    def load_weights(self, weights_path, weight_keys='params'):
        self.net = load_pretrained_network(self.net, weights_path, weight_keys=weight_keys)
    
    def forward(self, target, ref=None, **kwargs):
        device = self.dummy_param.device

        with torch.set_grad_enabled(self.as_loss):

            if self.metric_name == 'fid':
                output = self.net(target, ref, device=device, **kwargs)
            elif self.metric_name == 'inception_score':
                output = self.net(target, device=device, **kwargs)
            else:
                if not torch.is_tensor(target):
                    target = imread2tensor(target, rgb=True)
                    target = target.unsqueeze(0)
                    if self.metric_mode == 'FR':
                        assert ref is not None, 'Please specify reference image for Full Reference metric'
                        ref = imread2tensor(ref, rgb=True)
                        ref = ref.unsqueeze(0)

                if self.metric_mode == 'FR':
                    assert ref is not None, 'Please specify reference image for Full Reference metric'
                    output = self.net(target.to(device), ref.to(device), **kwargs)
                elif self.metric_mode == 'NR':
                    output = self.net(target.to(device), **kwargs)

        if self.as_loss:
            if isinstance(output, tuple):
                output = output[0]
            return weight_reduce_loss(output, self.loss_weight, self.loss_reduction)
        else:
            return output


def create_metric(metric_name, as_loss=False, device=None, **kwargs):
    assert metric_name in DEFAULT_CONFIGS.keys(), f'Metric {metric_name} not implemented yet.' 
    metric = CustomInferenceModel(metric_name, as_loss=as_loss, device=device, **kwargs)
    logger = get_root_logger()
    logger.info(f'Metric [{metric.net.__class__.__name__}] is created.')
    return metric


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def list_models(metric_mode=None, filter='', exclude_filters=''):
    """ Return list of available model names, sorted alphabetically
    Args:
        filter (str) - Wildcard filter string that works with fnmatch
        exclude_filters (str or list[str]) - Wildcard filters to exclude models after including them with filter
    Example:
        model_list('*ssim*') -- returns all models including 'ssim'
    """
    if metric_mode is None:
        all_models = DEFAULT_CONFIGS.keys()
    else:
        assert metric_mode in ['FR', 'NR'], f'Metric mode only support [FR, NR], but got {metric_mode}'
        all_models = [key for key in DEFAULT_CONFIGS.keys() if DEFAULT_CONFIGS[key]['metric_mode'] == metric_mode]

    if filter:
        models = []
        include_filters = filter if isinstance(filter, (tuple, list)) else [filter]
        for f in include_filters:
            include_models = fnmatch.filter(all_models, f)  # include these models
            if len(include_models):
                models = set(models).union(include_models)
    else:
        models = all_models
    if exclude_filters:
        if not isinstance(exclude_filters, (tuple, list)):
            exclude_filters = [exclude_filters]
        for xf in exclude_filters:
            exclude_models = fnmatch.filter(models, xf)  # exclude these models
            if len(exclude_models):
                models = set(models).difference(exclude_models)
    return list(sorted(models, key=_natural_key))
