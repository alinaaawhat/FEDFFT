import os
import torch
from federatedscope.core.aggregators import Aggregator
from federatedscope.core.auxiliaries.utils import param2tensor
from peft import FourierFTConfig

class ClientsAvgAggregator(Aggregator):
    """
    Implementation of vanilla FedAvg refer to 'Communication-efficient \
    learning of deep networks from decentralized data' [McMahan et al., 2017] \
    http://proceedings.mlr.press/v54/mcmahan17a.html
    """
    def __init__(self, model=None, device='cpu', config=None):
        super(Aggregator, self).__init__()
        self.model = model
        self.device = device
        self.cfg = config
        #cat [0-31]indices
        if str(self.model.model.active_peft_config).startswith("FourierFTConfig"):
            aggregated_indices = []
           
            for i in range(32):
                layer = self.model.model.base_model.model.model.layers[i]
                b_matrix = layer.self_attn.q_proj.indices['default']

                aggregated_indices.append(b_matrix)

            self.posit = torch.cat(aggregated_indices, dim=0)
            print(f'model is fft {self.posit.shape}')
            print(f'model is fft {self.posit}')

        else:
            print("model is not fft")

    def aggregate(self, agg_info):
        """
        To preform aggregation

        Arguments:
            agg_info (dict): the feedbacks from clients

        Returns:
            dict: the aggregated results
        """

        models = agg_info["client_feedback"]
        recover_fun = agg_info['recover_fun'] if (
            'recover_fun' in agg_info and self.cfg.federate.use_ss) else None
        avg_model = self._para_weighted_avg(models, recover_fun=recover_fun)
        return avg_model

    def update(self, model_parameters):
        """
        Arguments:
            model_parameters (dict): PyTorch Module object's state_dict.
        """
        self.model.load_state_dict(model_parameters, strict=False)

    def save_model(self, path, cur_round=-1):
        assert self.model is not None

        if self.cfg.llm.offsite_tuning.use and \
                self.cfg.llm.offsite_tuning.save_full_model:
            ckpt = {
                'cur_round': cur_round,
                'model': self.model.state_dict(return_trainable=False)
            }
        else:
            ckpt = {'cur_round': cur_round, 'model': self.model.state_dict()}
        torch.save(ckpt, path)

    def load_model(self, path):
        assert self.model is not None

        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model'])
            return ckpt['cur_round']
        else:
            raise ValueError("The file {} does NOT exist".format(path))

    def _para_weighted_avg(self, models, recover_fun=None):
        """
        Calculates the weighted average of models.
        """
        _, model_state_dict = models[0]  # 只取字典部分

        # check fourierft 
        has_fourierft = any("fourierft" in name for name in model_state_dict.keys())
        if has_fourierft:
            self.aggregation()  # 调用聚合函数
        
        
        #     # create global variable and T
        #     avg_position = torch.transpose(models[0][1]['posit'], 0, 1)
        #     # itera  local_model,create global position
        #     for sample_size, local_model in models:
        #         position = torch.transpose(local_model['posit'], 0, 1)
        #         # itera all position 
        #         for i, pos in enumerate(position):
        #             if pos not in avg_position:
        #                 avg_position = torch.cat((avg_position, pos.unsqueeze(1)), dim=1)
        #     # iterate avg_model every layers
        #     avg_model = {}
        #     for key in models[0][1]:
        #         avg_model[key] = []
        #         if key == 'posit':
        #             continue
        #         m = avg_position.shape[0] # position number
        #         global_sample_size = torch.ones(m).to('cuda:1') #
        #         avg_model[key] = torch.ones(m).to('cuda:1')
        #         # Average local model parameters
        #         for a in range(len(models)):
        #             sample_size, local_model = models[a]
        #             local_model[key] = param2tensor(local_model[key])
        #             local_position = torch.transpose(local_model['posit'], 0, 1)
        #             # iterate local_model position and calculate   and convert position to dictionary for lookup                        
        #             local_position_tuples = [tuple(pos.tolist()) for pos in local_position]
        #             avg_position_tuples = [tuple(pos.tolist()) for pos in avg_position]
        #             # local_position matches avg_position
        #             # print(f"find indices.......... {local_position_tuples.tolist()} {avg_position_tuples.tolist()}")
        #             for i, pos in enumerate(local_position_tuples):
        #                 if pos in avg_position_tuples:
        #                     index = avg_position_tuples.index(pos)
        #                     avg_value = avg_model[key][index].item()
        #                     local_value = local_model[key][i].item()
        #                     avg_model[key][index] = avg_value + local_value *sample_size
        #                     # avg_model[key][index] = avg_model[key][index].item() + local_model[key][i].item()*sample_size
        #                     global_sample_size[index] += sample_size
        #         # avg_model[key] = avg_model[key] / global_sample_size 
        #     avg_model['posit'] = avg_position        
        #     return avg_model
        else:
            print('Calculates the weighted average of models.')
            training_set_size = 0
            for i in range(len(models)):
                sample_size, _ = models[i]
                training_set_size += sample_size

            sample_size, avg_model = models[0]
            for key in avg_model:
                for i in range(len(models)):
                    local_sample_size, local_model = models[i]

                    if self.cfg.federate.ignore_weight:
                        weight = 1.0 / len(models)
                    elif self.cfg.federate.use_ss:
                        # When using secret sharing, what the server receives
                        # are sample_size * model_para
                        weight = 1.0
                    else:
                        weight = local_sample_size / training_set_size
                    if not self.cfg.federate.use_ss:
                        local_model[key] = param2tensor(local_model[key])
                    if i == 0:
                        avg_model[key] = local_model[key] * weight
                    else:
                        avg_model[key] += local_model[key] * weight
                if self.cfg.federate.use_ss and recover_fun:
                    avg_model[key] = recover_fun(avg_model[key])
                    # When using secret sharing, what the server receives are
                    # sample_size * model_para
                    avg_model[key] /= training_set_size
                    avg_model[key] = torch.FloatTensor(avg_model[key])
            return avg_model




class OnlineClientsAvgAggregator(ClientsAvgAggregator):
    """
    Implementation of online aggregation of FedAvg.
    """
    def __init__(self,
                 model=None,
                 device='cpu',
                 src_device='cpu',
                 config=None):
        super(OnlineClientsAvgAggregator, self).__init__(model, device, config)
        self.src_device = src_device

    def reset(self):
        """
        Reset the state of the model to its initial state
        """
        self.maintained = self.model.state_dict()
        for key in self.maintained:
            self.maintained[key].data = torch.zeros_like(
                self.maintained[key], device=self.src_device)
        self.cnt = 0

    def inc(self, content):
        """
        Increment the model weight by the given content.
        """
        if isinstance(content, tuple):
            sample_size, model_params = content
            for key in self.maintained:
                # if model_params[key].device != self.maintained[key].device:
                #    model_params[key].to(self.maintained[key].device)
                self.maintained[key] = (self.cnt * self.maintained[key] +
                                        sample_size * model_params[key]) / (
                                            self.cnt + sample_size)
            self.cnt += sample_size
        else:
            raise TypeError(
                "{} is not a tuple (sample_size, model_para)".format(content))

    def aggregate(self, agg_info):
        """
        Returns the aggregated value
        """
        return self.maintained
