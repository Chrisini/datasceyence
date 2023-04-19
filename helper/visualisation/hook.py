class Hook():
  # =============================================================================
  # Class to register a hook on the target layer (used to get the output channels of the layer)
  # https://github.com/juanigp/Pytorch-Deep-Dream/blob/master/Deep_Dream.ipynb

  # todo: https://pytorch.org/vision/stable/feature_extraction.html
  # =============================================================================

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()