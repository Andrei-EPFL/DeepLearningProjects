import matplotlib.pyplot as plt
import torch
import os
import argparse

from helpers import generate_pair_sets
from model import NN, train

if __name__ == '__main__':

    if torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'


    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', default=42, type=int)
    parser.add_argument('-nepochs', default=80, type=int)
    parser.add_argument('-test', default=False, action='store_true')
    args = parser.parse_args()
    
    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(5000)
    
    n_epochs=args.nepochs
    batch_size=5
    seed=args.seed

    model_fn = f"results/models/nn-seed{seed}.pt"

    if not os.path.isfile(model_fn) or not args.test:

        model = NN().to(device)   
    
        train_loss, val_loss, train_acc, val_acc = train(model, train_input, train_target, train_classes,
                test_input, test_target, test_classes, 
                n_epochs, batch_size, seed, device, validation_fraction=0.5, learning_rate=1e-3)

        
        torch.save(model.state_dict(), model_fn)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].plot(train_loss, label='train')
        ax[0].plot(val_loss, label='val')
        ax[0].set_ylabel('Loss')

        ax[1].plot(train_acc, label='train')
        ax[1].plot(val_acc, label='val')
        ax[1].set_ylabel('Accuracy')

        ax[0].legend(loc=0)
        [a.set_xlabel('Epochs') for a in ax]
        fig.savefig(f"results/plots/learning_curve.png", dpi=200)
    
    else:
        model = NN()
        model.load_state_dict(torch.load(model_fn))
        
    model.eval()

    out_1, out_2, out = model(test_input.to(device))
    out_classes = torch.argmax(out.to('cpu'), axis=1).to(int)
    class_1 = torch.argmax(out_1, axis=1)
    class_2 = torch.argmax(out_2, axis=1)

    accuracy = (out_classes == test_target).to(float).mean()
    print(f"Test accuracy = {accuracy}.")
    

    nrows = 5
    indices = torch.randint(low=0, high=test_input.shape[0], size = (nrows,))

    fig, ax = plt.subplots(2, nrows, figsize=((2*nrows, 2*2)))

    for i in range(nrows):
        id_ = indices[i]
        print(id_)
        ax[0, i].imshow(test_input[id_][0, :, :])
        ax[1, i].imshow(test_input[id_][1, :, :])
        
        ax[0, i].set_title(f"{class_1[id_]} <= {class_2[id_]}")
        ax[1, i].set_xlabel(f"Got {out_classes[id_]}. Expected {test_target[id_]}")

    fig.savefig(f"results/plots/test.png", dpi=200)
    
