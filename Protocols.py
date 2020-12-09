class Net:
    
    def __init__(self):
        self._layers    = []
        self._variables = []
        self._body      = None
        self._inference = None
        self._loss      = None
        self._updataOp  = None
        
    def body(self):
        pass
    
    def inference(self):
        pass
    
    def loss(self):
        pass
    
    def train(self):
        pass
    
    def evaluate(self):
        pass    
    
    @property
    def summary(self): 
        summs = []
        summs.append("=>Network Summary: ")
        for elem in self._layers:
            summs.append(elem.summary)
        summs.append("<=Network Summary: ")
        return "\n\n".join(summs)
    
