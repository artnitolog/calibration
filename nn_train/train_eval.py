from tqdm.auto import tqdm

def train_batch(model, x_batch, y_batch,
                loss_fn, optimizer):
    model.train()
    model.zero_grad()
    output = model(x_batch)
    loss = loss_fn(output, y_batch)
    loss.backward()
    optimizer.step()
    batch_loss = loss.cpu().item()
    with torch.no_grad():
        batch_acc_sum = (output.argmax(dim=1) == y_batch).sum().cpu().item()
    return batch_loss, batch_acc_sum

def train_epoch(model, dataloader, loss_fn, optimizer):
    epoch_loss = 0.0
    epoch_acc = 0
    epoch_size = 0
    for i_batch, (batch_x, batch_y) in enumerate(dataloader):
        batch_loss, batch_acc_sum = train_batch(model,
            batch_x.to(model.device), batch_y.to(model.device), loss_fn, optimizer)
        epoch_size += len(batch_x)
        epoch_loss += batch_loss * len(batch_x)
        epoch_acc += batch_acc_sum
        dataloader.set_postfix({'loss': f'{epoch_loss / epoch_size:.4f}',
                                'acc': f'{epoch_acc / epoch_size * 100:.2f}'})
    epoch_loss /= epoch_size
    epoch_acc /= epoch_size
    return epoch_loss, epoch_acc

def train_full(model, train_dataloader, val_dataloader,
               loss_fn, optimizer, scheduler,
               n_epochs, callback=None):
    epochs = tqdm(range(n_epochs), desc='Epochs', leave=True)
    for i_epoch in epochs:
        tqdm_dataloader = tqdm(train_dataloader, leave=True, desc='batches')
        epoch_loss, epoch_acc = train_epoch(model, tqdm_dataloader, loss_fn, optimizer)
        if callback is not None:
            cb_dict = callback(model, val_dataloader, loss_fn, epoch_loss, epoch_acc)
            epochs.set_postfix(cb_dict)
        scheduler.step()
        
def eval_calib(model, dataloader, loss_fn):
    model.eval()
    logits = []
    targets = []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            output = model(batch_x.to(model.device)).cpu()
            logits.append(output)
            targets.append(batch_y)
    logits = torch.cat(logits)
    targets = torch.cat(targets)
    loss = loss_fn(logits, targets).item()
    acc = (logits.argmax(dim=1) == targets).sum().item() / len(targets)
    return loss, acc

class CallBack:
    def __init__(self, eval_fn, name=None):
        self.eval_fn = eval_fn
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
    
    def last_info(self):
        return {'loss_train': f'{self.train_losses[-1]:.4f}',
                'acc_train': f'{self.train_accs[-1] * 100:.2f}',
                'loss_val': f'{self.val_losses[-1]:.4f}',
                'acc_val': f'{self.val_accs[-1] * 100:.2f}'}

    def __call__(self, model, val_dataloader, loss_fn,
                 epoch_loss=None, epoch_acc=None):
        self.train_losses.append(epoch_loss)
        self.train_accs.append(epoch_acc)
        tqdm_loader = tqdm(val_dataloader, desc='eval (val)')
        loss_val, acc_val = self.eval_fn(model, tqdm_loader, loss_fn)
        self.val_losses.append(loss_val)
        self.val_accs.append(acc_val)
        tqdm_loader.set_postfix({'loss': f'{loss_val:.4f}',
                                 'acc': f'{acc_val * 100:.2f}'})
        return self.last_info()
