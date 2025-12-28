from accelerate.utils import is_peft_model
from torch import nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import PreTrainedModel


class DualModel(nn.Module):
    def __init__(
        self,
        leader: PreTrainedModel,
        follower: PreTrainedModel,
        use_follower_for_forward: bool = False,
    ):
        super().__init__()
        # Store the two HF models
        self.leader = leader
        self.follower = follower
        # A flag if you want forward() to invoke follower instead of leader
        self.use_follower = use_follower_for_forward
        self.warnings_issued = {}
        self.config = leader.config
        self.generation_config = leader.config

    def switch_to_leader(self):
        self.use_follower = False

    def switch_to_follower(self):
        self.use_follower = True

    def to(self, *args, **kwargs):
        """
        Ensure .to() moves both sub‑models.
        """
        self.leader = self.leader.to(*args, **kwargs)
        self.follower = self.follower.to(*args, **kwargs)
        return self

    # @property
    # def config(self):
    #     return self.follower.config if self.use_follower else self.leader.config

    @property
    def device(self):
        return self.follower.device if self.use_follower else self.leader.device

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        By default runs the leader; if `self.use_follower` is True, runs follower.
        You could also split the batch between them, or return both outputs, etc.
        """
        if self.use_follower:
            return self.follower(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )
        else:
            return self.leader(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )

    def generate(self, *args, use_follower: bool = False, **kwargs):
        """
        Proxy generate calls. Pass use_follower=True to invoke follower.generate()
        """
        if use_follower:
            return self.follower.generate(*args, **kwargs)
        else:
            return self.leader.generate(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """
        Proxy for generation preprocessing (beam search, sampling, etc).
        """
        m = self.follower if self.use_follower else self.leader
        return m.prepare_inputs_for_generation(*args, **kwargs)


def switch_model(model, adapter_name):
    """
    Switch the PEFT adapter of the model to the specified adapter name.
    """
    assert adapter_name in ["leader", "follower"]
    if is_ddp_wrapped(model):
        if is_peft_model(model.module) and set(model.module.peft_config.keys()) == {
            "leader",
            "follower",
        }:
            model.module.set_adapter(adapter_name)
        else:
            if adapter_name == "leader":
                model.module.switch_to_leader()
            else:
                model.module.switch_to_follower()
    else:
        if is_peft_model(model) and set(model.peft_config.keys()) == {
            "leader",
            "follower",
        }:
            model.set_adapter(adapter_name)
        else:
            if adapter_name == "leader":
                model.switch_to_leader()
            else:
                model.switch_to_follower()


def is_ddp_wrapped(model):
    return isinstance(model, DDP)
