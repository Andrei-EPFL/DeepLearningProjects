import matplotlib.pyplot as plt
import torch
import os
import argparse

from helpers import generate_pair_sets
from model import train_bline, ConvNet

if __name__ == '__main__':

    if torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'


    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', default=42, type=int)
    parser.add_argument('-nepochs', default=50, type=int)
    parser.add_argument('-test', default=False, action='store_true')
    parser.add_argument('-single', default=False, action='store_true')
    

    args = parser.parse_args()
    torch.manual_seed(999) # Set seed because generate_pair_sets involves randomness
    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(2000)
    

    train_input = train_input[:2000]
    train_target = train_target[:2000]
    train_classes = train_classes[:2000]
    
    n_epochs=args.nepochs
    batch_size=5
    seed=args.seed

    
    if args.single:

        model_fn = f"results/models/baseline-seed{seed}.pt"
        if (not os.path.isfile(model_fn) or not args.test):
            torch.manual_seed(seed)
            model = ConvNet(in_channels=2, n_classes=2).to(device)
                
            train_loss, val_loss, train_acc, val_acc = train_bline(model, train_input, train_target, train_classes,
                    n_epochs, batch_size, device, validation_fraction=0.5, learning_rate=1e-4)

            
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
            fig.savefig(f"results/plots/learning_curve_seed{seed}-baseline.png", dpi=200)
        
        else:
            model = ConvNet(in_channels=2, n_classes=2)
            model.load_state_dict(torch.load(model_fn))
            
        model.eval()

        out = model(test_input)
        out_classes = torch.argmax(out.to('cpu'), axis=1).to(int)
        
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
            ax[1, i].set_xlabel(f"Got {out_classes[id_]}. Expected {test_target[id_]}")

        fig.savefig(f"results/plots/test-seed{seed}-baseline.png", dpi=200)
    

    else:
        seeds = range(15)
        test_accuracies = []
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        for seed in seeds:
            model_fn = f"results/models/baseline-seed{seed}.pt"
            
            if (not os.path.isfile(model_fn) or not args.test):
            
                torch.manual_seed(seed)
                model = ConvNet(in_channels=2, n_classes=2).to(device)
            
                train_loss, val_loss, train_acc, val_acc = train_bline(model, train_input, train_target, train_classes,
                        n_epochs, batch_size, device, validation_fraction=0.5, learning_rate=1e-4)

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
                fig.savefig(f"results/plots/learning_curve_seed{seed}-baseline.png", dpi=200)

            else:
                model = ConvNet(in_channels=2, n_classes=2)
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

        if args.test:
            train_losses = torch.load(f"results/data/ensemble_train_loss-baseline.pkl")
            val_losses = torch.load(f"results/data/ensemble_val_loss-baseline.pkl")
            train_accs = torch.load(f"results/data/ensemble_train_acc-baseline.pkl")
            val_accs = torch.load(f"results/data/ensemble_val_acc-baseline.pkl")
        else:
            train_losses = torch.stack(train_losses)
            val_losses = torch.stack(val_losses)
            train_accs = torch.stack(train_accs)
            val_accs = torch.stack(val_accs)
            torch.save(train_losses, f"results/data/ensemble_train_loss-baseline.pkl")
            torch.save(val_losses, f"results/data/ensemble_val_loss-baseline.pkl")
            torch.save(train_accs, f"results/data/ensemble_train_acc-baseline.pkl")
            torch.save(val_accs, f"results/data/ensemble_val_acc-baseline.pkl")
            
        
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
        fig.savefig(f"results/plots/learning_curve_ensemble-baseline.png", dpi=200)

          

            