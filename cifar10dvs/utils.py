

class CoefficientScheduler:
    def __init__(self, model):
        # 统计 model 中有多少层
        self.model = model
        self.cnt = 0

    def count_neuron_layers(self):
        for m in self.model:
            if hasattr(m, "surrogate_function"):
                self.cnt += 1
