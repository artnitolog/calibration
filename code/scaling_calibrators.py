from torch.nn.functional import cross_entropy
from scipy.special import softmax


def tt(np_array):
    return torch.from_numpy(np_array)


class LogitScaling:
    def __init__(self, scale_type='temperature', bias_type='none'):
        '''
        scale_type: str:
            'temperature': single-parameter logits' scaling
            'vector': vector scaling (logits multiplied by diagonal matrix)
            'matrix': matrix scaling
        bias_type: str
            'intercept': one bias (number) for all classes
            'vector': individual biases for each class
            'none': no bias term
        '''
        self.device = device
        self.scale_type = scale_dim
        self.bias_type = bias_type
        self.scale = None
        self.bias = None
    
    def fit(self, logits_val, targets_val, device='cpu', lr=1.0, max_iter=1000):
        logits_cal = tt(logits_val).to(device)
        targets_cal = tt(targets_val).to(device)
        kwargs = {'dtype': logits_cal.dtype, 'requires_grad': True, 'device': device}
        if scale_type == 'temperature':
            scale = torch.tensor(1.0, **kwargs)
        elif scale_type == 'vector':
            scale = torch.ones(logits_cal.shape[1], **kwargs)
        elif scale_type == 'matrix':
            scale = torch.eye(logits_cal.shape[1], **kwargs)
        params = [scale]
        if self.bias_type == 'intercept':
            bias = torch.tensor(0.0, **kwargs)
            params.append(bias)
        elif bias_type == 'vector':
            bias = torch.zeros(logits_cal.shape[1], **kwargs)
            params.append(bias)
        else:
            bias = torch.tensor(0, requires_grad=False, dtype=logits_cal.dtype, device=device)
            
        optimizer = torch.optim.LBFGS(params, lr=lr, max_iter=max_iter)
        def closure():
            optimizer.zero_grad()
            if self.scale_type == 'matrix':
                loss = cross_entropy(logits_cal @ scale + bias, targets_cal)
            else:
                loss = cross_entropy(logits_cal * scale + bias, targets_cal)
            loss.backward()
            return loss
        optimizer.step(closure)
        self.n_iter = optimizer.state[vector]['n_iter']
        self.scale = scale.detach().cpu().numpy()
        self.bias = bias.detach().cpu().numpy()
    
    def transform(self, logits_test):
        if self.scale_type == 'matrix':
            return softmax(logits_test @ self.scale + self.bias, axis=1)
        else:
            return softmax(logits_test * self.scale + self.bias, axis=1)
