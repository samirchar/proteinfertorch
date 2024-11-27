
class ExponentialDecay:
    def __init__(self, decay_steps, decay_rate, staircase):
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def __call__(self, step):
        if self.staircase:
            return self.decay_rate ** (step // self.decay_steps)
        else:
            return self.decay_rate ** (step / self.decay_steps)

