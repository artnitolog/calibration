import torch
from torch.nn.functional import cross_entropy
from torch.nn.functional import binary_cross_entropy_with_logits
from scipy.special import softmax
from scipy.special import expit as sigmoid


class LogitScaling:
    def __init__(self, scale_type='temperature', bias_type='none'):
        '''
        scale_type: str
            'temperature': single-parameter logits' scaling
            'vector': vector scaling (logits multiplied by diagonal matrix)
            'matrix': matrix scaling
            'platt': classic Platt Scaling mode, this automatically sets
                     bias_type to 'intercept'

        bias_type: str
            'intercept': one bias (number) for all classes
            'vector': individual biases for each class
            'none': no bias term
        '''
        self.scale_type = scale_type
        if scale_type == 'platt':
            self.bias_type = 'intercept'
        else:
            self.bias_type = bias_type
        self.scale = None
        self.bias = None
        self.n_iter = None
    
    def fit(self, logits_cal, targets_cal, device='cpu', lr=1.0, max_iter=1000):
        '''
        Args:
            logits_cal: np.array (n, n_classes)
            targets_cal: np.array (n_classes,)
            device: which device to use for optimization
            lr: learning rate of optimizer
            max_iter: maximal number of iterations
        '''
        logits_cal = torch.from_numpy(logits_cal).to(device)
        targets_cal = torch.from_numpy(targets_cal).to(device)
        if self.scale_type == 'platt':  # dtype for binary CE
            targets_cal = targets_cal.to(dtype=logits_cal.dtype)
        kwargs = {'dtype': logits_cal.dtype,
                  'device': device,
                  'requires_grad': True}
        if self.scale_type == 'temperature' or self.scale_type == 'platt':
            scale = torch.tensor(1.0, **kwargs)
        elif self.scale_type == 'vector':
            scale = torch.ones(logits_cal.shape[1], **kwargs)
        elif self.scale_type == 'matrix':
            scale = torch.eye(logits_cal.shape[1], **kwargs)
        params = [scale]
        if self.bias_type == 'intercept':
            bias = torch.tensor(0.0, **kwargs)
            params.append(bias)
        elif self.bias_type == 'vector':
            bias = torch.zeros(logits_cal.shape[1], **kwargs)
            params.append(bias)
        else:
            bias = torch.tensor(0.0, requires_grad=False, dtype=logits_cal.dtype, device=device)
            
        optimizer = torch.optim.LBFGS(params, lr=lr, max_iter=max_iter)
        def closure():
            optimizer.zero_grad()
            if self.scale_type == 'platt':
                loss = binary_cross_entropy_with_logits(logits_cal * scale + bias, targets_cal)
            elif self.scale_type == 'matrix':
                loss = cross_entropy(logits_cal @ scale + bias, targets_cal)
            else:
                loss = cross_entropy(logits_cal * scale + bias, targets_cal)
            loss.backward()
            return loss
        optimizer.step(closure)
        self.n_iter = optimizer.state[scale]['n_iter']
        self.scale = scale.detach().cpu().numpy()
        self.bias = bias.detach().cpu().numpy()
        return self
    
    def transform(self, logits_test):
        '''
        Args:
            logits_cal: np.array (n, n_classes)
        '''
        if self.scale_type == 'platt':
            return sigmoid(logits_test * self.scale + self.bias)
        elif self.scale_type == 'matrix':
            return softmax(logits_test @ self.scale + self.bias, axis=1)
        else:
            return softmax(logits_test * self.scale + self.bias, axis=1)
