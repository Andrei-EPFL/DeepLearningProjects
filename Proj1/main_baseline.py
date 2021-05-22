import matplotlib.pyplot as plt
import torch
import os
import argparse

from helpers import generate_pair_sets
from train import train_bline
from convnet import ConvNet, Baseline
from resnet import ResNet_Baseline

if __name__ == '__main__':

    if torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'


    parser = argparse.ArgumentParser()
    parser.add_argument('-seeds', default=[42], type=int, nargs='+')
    parser.add_argument('-nepochs', default=50, type=int)
    parser.add_argument('-model', default='convnet', type=str)
    parser.add_argument('-test', default=False, action='store_true')
    
    
    

    args = parser.parse_args()
    torch.manual_seed(999) # Set seed because generate_pair_sets involves randomness
    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(2000)
    

    train_input = train_input[:2000]
    train_target = train_target[:2000]
    train_classes = train_classes[:2000]
    
    n_epochs=args.nepochs
    batch_size=5
    

    
    seeds = args.seeds
    test_accuracies = []
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    for seed in seeds:
        model_fn = f"results/{args.model}/models/baseline-seed{seed}.pt"
        
        if (not os.path.isfile(model_fn) or not args.test):
        
            torch.manual_seed(seed)
            if args.model == 'convnet':
                model = Baseline().to(device)
            elif args.model == 'resnet':
                model = ResNet_Baseline().to(device)
            else:
                raise ValueError("-model must be either convnet or resnet")
        
            train_loss, val_loss, train_acc, val_acc = train_bline(model, train_input, train_target, train_classes,
                    n_epochs, batch_size, device, validation_fraction=0.5, learning_rate=1e-3)

            train_losses.append(torch.Tensor(train_loss))
            val_losses.append(torch.Tensor(val_loss))
            train_accs.append(torch.Tensor(train_acc))
            val_accs.append(torch.Tensor(val_acc))
            torch.save(model.state_dict(), model_fn)
            model.to('cpu')
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            ax[0].plot(train_loss, label='train')
            ax[0].plot(val_loss, label='val')
            ax[0].set_ylabel('Loss')

            ax[1].plot(train_acc, label='train')
            ax[1].plot(val_acc, label='val')
            ax[1].set_ylabel('Accuracy')

            ax[0].legend(loc=0)
            [a.set_xlabel('Epochs') for a in ax]
            fig.savefig(f"results/{args.model}/plots/learning_curve_seed{seed}-baseline.png", dpi=200)

        else:
            if args.model == 'convnet':
                model = Baseline()
            elif args.model == 'resnet':
                model = ResNet_Baseline()
            else:
                raise ValueError("-model must be either convnet or resnet")
            model.load_state_dict(torch.load(model_fn))



        model.eval()

        #indices = torch.randperm(test_input.shape[0])
        indices = torch.arange(test_input.shape[0])
        out = model(test_input[indices[:1000]])

        out_classes = torch.argmax(out.to('cpu'), axis=1)

        accuracy = (out_classes == test_target[indices[:1000]]).to(float).mean()
        test_accuracies.append(accuracy)
        print(f"Test accuracy = {accuracy}.")

    test_accuracies = torch.Tensor(test_accuracies)
    print(f"Average test accuracy = {torch.mean(test_accuracies)} +/- {torch.std(test_accuracies)}")
    if len(args.seeds) > 1:
        if args.test:
            train_losses = torch.load(f"results/{args.model}/data/ensemble_train_loss-baseline.pkl")
            val_losses = torch.load(f"results/{args.model}/data/ensemble_val_loss-baseline.pkl")
            train_accs = torch.load(f"results/{args.model}/data/ensemble_train_acc-baseline.pkl")
            val_accs = torch.load(f"results/{args.model}/data/ensemble_val_acc-baseline.pkl")
        else:
            train_losses = torch.stack(train_losses)
            val_losses = torch.stack(val_losses)
            train_accs = torch.stack(train_accs)
            val_accs = torch.stack(val_accs)
            torch.save(train_losses, f"results/{args.model}/data/ensemble_train_loss-baseline.pkl")
            torch.save(val_losses, f"results/{args.model}/data/ensemble_val_loss-baseline.pkl")
            torch.save(train_accs, f"results/{args.model}/data/ensemble_train_acc-baseline.pkl")
            torch.save(val_accs, f"results/{args.model}/data/ensemble_val_acc-baseline.pkl")
            
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        x = torch.arange(args.nepochs)
        ax[0].errorbar(x, train_losses.mean(axis=0), yerr = train_losses.std(axis=0), label='train')
        ax[0].errorbar(x, val_losses.mean(axis=0), yerr = val_losses.std(axis=0), label='val')
        ax[0].set_ylabel('Loss')

        ax[1].errorbar(x, train_accs.mean(axis=0), yerr = train_accs.std(axis=0), label='train')
        ax[1].errorbar(x, val_accs.mean(axis=0), yerr = val_accs.std(axis=0), label='val')
        ax[1].set_ylabel('Accuracy')

        ax[0].legend(loc=0)
        [a.set_xlabel('Epochs') for a in ax]
        fig.savefig(f"results/{args.model}/plots/learning_curve_ensemble-baseline.png", dpi=200)

            

            
