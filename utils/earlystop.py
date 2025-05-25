import math

class EarlyStopping():
    def __init__(self, monitor: str = 'val_loss', mode: str = 'min', patience: int = 1):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.__value = math.inf if mode == 'min' else -math.inf
        self.__times = 0

    def state_dict(self) -> dict:
        return {
            'monitor': self.monitor,
            'mode': self.mode,
            'patience': self.patience,
            'value': self.__value,
            'times': self.__times
        }

    def load_state_dict(self, state_dict: dict):
        self.monitor = state_dict['monitor']
        self.mode = state_dict['mode']
        self.patience = state_dict['patience']
        self.__value = state_dict['value']
        self.__times = state_dict['times']

    def reset(self):
        self.__times = math.inf if self.mode == 'min' else -math.inf

    def __call__(self, metrics):
        best = False
        print("当前最优值:", self.__value, ",当前耐心值:", self.__times, ",当前值:", metrics)
        if (self.mode == 'min' and metrics <= self.__value) or (
                self.mode == 'max' and metrics >= self.__value):
            # 如果是最优则更新数据
            self.__value = metrics
            self.__times = 0
            best = True
        else:
            # 如果不是最优则加一
            self.__times += 1
            best = False
            
        # 如果超过了耐心值则返回 True
        if self.__times >= self.patience:
            return True, best
        # 如果没有超过耐心值则返回 False
        return False, best