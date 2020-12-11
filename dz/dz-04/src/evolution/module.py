class Module:
    def apply(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def __str__(self):
        return "evolution.Module"
