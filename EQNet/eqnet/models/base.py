"""
The basic function for all modules defined in:
    * query_stage.
    * support processor.
    * support producer.
"""

from torch import nn
from eqnet.utils.config import cfg, cfg_from_yaml_file, merge_new_config


class BasicEQModules(nn.Module):
    def __init__(self, model_cfg, adapt_model_cfg):
        super().__init__()
        if isinstance(model_cfg, str):
            # input a configure file.
            cfg_from_yaml_file(model_cfg, cfg)
            model_cfg = cfg

        model_cfg = merge_new_config(model_cfg, adapt_model_cfg)
        self.model_cfg = model_cfg

    def _parse_input_dict(self, data_dict):
        """ A function for specifically preprocessing inputs from different codebases.

        e.g.: Map the keys from different codebases to the demanded keys in _forward_input_dict.
        """
        input_dict = dict()
        return input_dict

    def _forward_input_dict(self, input_dict):
        """ Main forward function
        """
        output_dict = dict()
        return output_dict

    def _parse_output_dict(self, output_dict):
        """ A function for mapping keys in output_dict to the keys required by different codebases.
        """
        data_dict = dict()
        return data_dict

    def forward(self, data_dict):
        """ Define forward pipeline.
        """
        input_dict = self._parse_input_dict(data_dict)
        output_dict = self._forward_input_dict(input_dict)
        new_data_dict = self._parse_output_dict(output_dict)

        if isinstance(data_dict, dict):
            data_dict.update(new_data_dict)
        else:
            data_dict = new_data_dict
        return data_dict
