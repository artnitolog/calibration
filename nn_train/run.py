import torch
import torch.nn as nn
import torch.optim as optim

train_dataloader, val_dataloader = cifar_train_val(version=100, num_workers=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = LeNet(num_classes=100)
# model = ResNet50(num_classes=100)
model = MobileNetV2(num_classes=100)
# model = EfficientNetB0(num_classes=100)

model.to(device)
model.device = device

loss_fn = nn.CrossEntropyLoss()
cb = CallBack(eval_calib)
optimizer = optim.SGD(model.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=120)

train_full(model, train_dataloader, val_dataloader,
            loss_fn, optimizer, scheduler,
            n_epochs=360, callback=cb)
