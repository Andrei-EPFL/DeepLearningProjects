import torch
from helpers import generate_pair_sets
from convnet import NN_ws
from train import train

DIGITS = torch.arange(0, 10)


if torch.cuda.is_available():
    device='cuda'
else:
    device='cpu'


train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(2000)

n_epochs=50
batch_size=5


seed = 42


torch.manual_seed(seed)
model = NN_ws().to(device)   

train_loss, val_loss, train_acc, val_acc = train(model, train_input, train_target, train_classes,
                    n_epochs, batch_size, device, validation_fraction=0.5, learning_rate=1e-3, use_aux_loss=True)

model.eval()
criterion = torch.nn.CrossEntropyLoss().to(device)
test_input = test_input.to(device)
test_target = test_target.to(device)
test_classes = test_classes.to(device)

indices = torch.arange(test_input.shape[0]) # indexing, could be randomized instead
out_1, out_2, out = model(test_input[indices[:1000]])

loss_out = criterion(out, test_target[indices[:1000]])
loss_1 = criterion(out_1, test_classes[indices[:1000],0])
loss_2 = criterion(out_2, test_classes[indices[:1000],1])
test_loss = (loss_out + loss_1 + loss_2).item()
print(f"Test loss = {test_loss:.3e}")


out_classes = torch.argmax(out, axis=1)
class_1 = torch.argmax(out_1, axis=1)
class_2 = torch.argmax(out_2, axis=1)
accuracy_1 = (class_1 == test_classes[indices[:1000], 0]).to(float).mean()
accuracy_2 = (class_2 == test_classes[indices[:1000], 1]).to(float).mean()
out_class_comp = (DIGITS[class_1] <= DIGITS[class_2]).to(int).to(device)
accuracy = (out_classes == test_target[indices[:1000]]).to(float).mean()
accuracy_comp = (out_class_comp == test_target[indices[:1000]]).to(float).mean()


print(f"Test accuracy = {accuracy:.3f}.")
print(f"Test accuracy digit prediction comparison = {accuracy_comp:.3f}.")
print(f"Test digit class accuracies = {accuracy_1:.3f}, {accuracy_2:.3f}.")
