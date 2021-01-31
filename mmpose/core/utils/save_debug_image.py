from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class SaveDebugImage(Hook):

    def __init__(self):
        pass

    def before_run(self, runner):
        print('before_run')
        pass

    def after_run(self, runner):
        print('after_run')
        pass

    def before_epoch(self, runner):
        print('before_epoch')
        pass

    def after_epoch(self, runner):
        print('after_epoch')
        pass

    def before_iter(self, runner):
        print('before_iter')
        pass

    def after_iter(self, runner):
        print('after_iter')
        pass
