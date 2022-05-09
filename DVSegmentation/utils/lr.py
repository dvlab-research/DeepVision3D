from torch.optim.lr_scheduler import LambdaLR, StepLR, OneCycleLR, MultiStepLR

class LambdaStepLR(LambdaLR):

  def __init__(self, optimizer, lr_lambda, last_step=-1):
    super(LambdaStepLR, self).__init__(optimizer, lr_lambda, last_step)

  @property
  def last_step(self):
    """Use last_epoch for the step counter"""
    return self.last_epoch

  @last_step.setter
  def last_step(self, v):
    self.last_epoch = v


class PolyLR(LambdaStepLR):
  """DeepLab learning rate policy"""

  def __init__(self, optimizer, max_iter, power=0.9, last_step=-1):
    super(PolyLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1))**power, last_step)


class SquaredLR(LambdaStepLR):
  """ Used for SGD Lars"""

  def __init__(self, optimizer, max_iter, last_step=-1):
    super(SquaredLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1))**2, last_step)


class ExpLR(LambdaStepLR):

  def __init__(self, optimizer, step_size, gamma=0.9, last_step=-1):
    super(ExpLR, self).__init__(optimizer, lambda s: gamma**(s / step_size), last_step)


def initialize_scheduler(optimizer, config, last_epoch=-1, scheduler_epoch=True, logger=None):
  last_step = -1 if last_epoch < 0 else config.iter_per_epoch_train * (last_epoch + 1) - 1
  if scheduler_epoch:
    if 'step_size' in config:
      config.step_size = config.iter_per_epoch_train * config.step_size
    if 'exp_step_size' in config:
      config.exp_step_size = config.iter_per_epoch_train * config.exp_step_size

  if config.scheduler == 'StepLR':
    return StepLR(optimizer, step_size=config.step_size, gamma=config.step_gamma, last_epoch=last_step)
  elif config.scheduler == 'MultiStepLR':
    return MultiStepLR(optimizer, milestones=config.lr_decay_epochs, gamma=config.step_gamma, last_epoch=last_step)
  elif config.scheduler == 'PolyLR':
    return PolyLR(optimizer, max_iter=config.max_iter, power=config.poly_power, last_step=last_step)
  elif config.scheduler == 'SquaredLR':
    return SquaredLR(optimizer, max_iter=config.max_iter, last_step=last_step)
  elif config.scheduler == 'ExpLR':
    return ExpLR(optimizer, step_size=config.exp_step_size, gamma=config.exp_gamma, last_step=last_step)
  elif config.scheduler == 'OneCycleLR':
    return OneCycleLR(optimizer, max_lr=config.oc_max_lr, total_steps=config.max_iter, pct_start=config.oc_pct_start,
                      anneal_strategy=config.oc_anneal_strategy, div_factor=config.oc_div_factor,
                      final_div_factor=config.oc_final_div_factor, last_epoch=last_step)
  else:
    if logger is not None:
      logger.info('Scheduler not supported')
    else: print('Scheduler not supported')