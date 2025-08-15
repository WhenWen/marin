from levanter.optim import MudamConfig
from typing import Optional


def mudam_config(
    learning_rate: Optional[float] = None,
    adam_lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    beta1: Optional[float] = None,
    beta2: Optional[float] = None,
    shampoo_beta: Optional[float] = None,
    epsilon: Optional[float] = None,
    max_grad_norm: Optional[float] = None,
    warmup: Optional[float] = None,
    decay: Optional[float] = None,
    lr_schedule: Optional[str] = None,
    momentum: Optional[float] = None,
    momentum_2: Optional[float] = None,
    min_lr_ratio: Optional[float] = None,
    cycle_length: Optional[int] = None,
    mu_dtype: Optional[str] = None,
    precond_dtype: Optional[str] = None,
    normalization: Optional[str] = None,
    another_muon: Optional[bool] = None,
) -> MudamConfig:
    optimizer=MudamConfig(
            learning_rate=learning_rate,
            adam_lr=(
                adam_lr if adam_lr is not None else MudamConfig().adam_lr
            ),
            weight_decay=(
                weight_decay if weight_decay is not None else MudamConfig().weight_decay
            ),
            beta1=(beta1 if beta1 is not None else MudamConfig().beta1),
            beta2=(beta2 if beta2 is not None else MudamConfig().beta2),
            shampoo_beta=(
                shampoo_beta if shampoo_beta is not None else MudamConfig().shampoo_beta
            ),
            momentum=(momentum if momentum is not None else MudamConfig().momentum),
            momentum_2=(momentum_2 if momentum_2 is not None else MudamConfig().momentum_2),
            epsilon=(epsilon if epsilon is not None else MudamConfig().epsilon),
            max_grad_norm=(
                max_grad_norm if max_grad_norm is not None else MudamConfig().max_grad_norm
            ),
            warmup=(warmup if warmup is not None else MudamConfig().warmup),
            decay=(decay if decay is not None else MudamConfig().decay),
            lr_schedule=(lr_schedule if lr_schedule is not None else MudamConfig().lr_schedule),
            cycle_length=cycle_length,  # can be int, list[int], or None
            min_lr_ratio=(
                min_lr_ratio if min_lr_ratio is not None else MudamConfig().min_lr_ratio
            ),
            mu_dtype=(mu_dtype if mu_dtype is not None else MudamConfig().mu_dtype),
            precond_dtype=(precond_dtype if precond_dtype is not None else MudamConfig().precond_dtype),
            normalization=normalization,
            another_muon=(another_muon if another_muon is not None else MudamConfig().another_muon)
        )
        
    return optimizer