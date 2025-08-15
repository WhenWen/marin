import hashlib

from marin.execution.executor import unwrap_versioned_value


def mudam_train_config(prefix: str, config):
    name = (
        f"lr{unwrap_versioned_value(config.optimizer_config.learning_rate)}-"
        f"alr{unwrap_versioned_value(config.optimizer_config.adam_lr)}-"
        f"wd{unwrap_versioned_value(config.optimizer_config.weight_decay)}-"
        f"minlr{unwrap_versioned_value(config.optimizer_config.min_lr_ratio)}-"
        f"warmup{unwrap_versioned_value(config.optimizer_config.warmup)}-"
        f"sb1{unwrap_versioned_value(config.optimizer_config.momentum)}-"
        f"sb2{unwrap_versioned_value(config.optimizer_config.momentum_2)}-"
        f"b1{unwrap_versioned_value(config.optimizer_config.beta1)}-"
        f"b2{unwrap_versioned_value(config.optimizer_config.beta2)}-"
        f"sb{unwrap_versioned_value(config.optimizer_config.shampoo_beta)}-"
        f"gn{unwrap_versioned_value(config.optimizer_config.max_grad_norm)}-"
        f"steps{unwrap_versioned_value(config.num_train_steps)}"
        f"eps{unwrap_versioned_value(config.optimizer_config.epsilon)}-"
        f"norm{unwrap_versioned_value(config.optimizer_config.normalization)}"
        f"amu{unwrap_versioned_value(config.optimizer_config.another_muon)}"
    )
    first_hash = hashlib.md5(name.encode()).hexdigest()[:6]
    return (prefix + first_hash + name)[:64]
