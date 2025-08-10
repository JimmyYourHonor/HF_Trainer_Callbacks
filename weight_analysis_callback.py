import torch
import torch.nn as nn
from transformers.integrations import WandbCallback
from .utils import AverageMeter


class WeightAnalysisCallback(WandbCallback):
    """
    A callback to analyze the weights of the model during training.
    And reports to Weights & Biases (wandb).
    """

    def on_train_begin(self, args, state, control, **kwargs):
        self.update_ratios_avg = {}
        self.grad_ratios_avg = {}
        self.previous_update = {}
        self.update_smoothness = []
        self.update_ratios_avgs = []
        self.grad_ratios_avgs = []
        super().on_train_begin(args, state, control, **kwargs)

    def on_epoch_begin(self, args, state, control, **kwargs):
        model = kwargs['model']
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.update_ratios_avg[name] = AverageMeter()
                self.grad_ratios_avg[name] = AverageMeter()

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self.params_before = {}
        self.grads = {}
        model = kwargs['model']
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.params_before[name] = param.detach().cpu()
                self.grads[name] = param.grad.detach().cpu() if param.grad is not None else None

    def on_optimizer_step(self, args, state, control, **kwargs):
        self.params_after = {}
        model = kwargs['model']
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.params_after[name] = param.detach().cpu()

        if self.previous_update:
            self.update_smoothness.append({})
        for name in self.params_before.keys():
            if name in self.params_after:
                update = self.params_after[name] - self.params_before[name]
                grad = self.grads[name]
                if grad is not None and grad.norm() > 0:
                    update_ratio = (update.norm() / self.params_before[name].norm()).log10().data.item()
                    grad_ratio = nn.functional.cosine_similarity(grad.flatten(), update.flatten(), dim=0).data.item()
                    self.update_ratios_avg[name].update(update_ratio, weight=1)
                    self.grad_ratios_avg[name].update(grad_ratio, weight=1)
            # Store the smoothness of updates
            if name in self.previous_update:
                prev_update = self.previous_update[name]
                angle = torch.acos(torch.dot(update.flatten(), prev_update.flatten())) / \
                (torch.linalg.norm(update) * torch.linalg.norm(prev_update))
                if len(self.update_smoothness) == 1:
                    self.update_smoothness[0][name] = angle
                else:
                    acc_angle = self.update_smoothness[-2][name] + angle
                    self.update_smoothness[-1][name] = acc_angle
                    self._wandb.log({
                        f'update_smoothness/{name}' : acc_angle
                    })
            self.previous_update[name] = update

    def on_epoch_end(self, args, state, control, **kwargs):
        self.grad_ratios_avgs.append({})
        self.update_ratios_avgs.append({})
        for name in self.grad_ratios_avg.keys():
            self.grad_ratios_avgs[-1][name] = self.grad_ratios_avg[name].average()
            self.update_ratios_avgs[-1][name] = self.update_ratios_avg[name].average()
            self._wandb.log({
                f'update_ratios/{name}': self.update_ratios_avg[name].average(),
                f'grad_ratios/{name}': self.grad_ratios_avg[name].average(),
            })