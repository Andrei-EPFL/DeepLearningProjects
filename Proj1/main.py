import matplotlib.pyplot as plt
import torch
import os
import argparse

from helpers import generate_pair_sets

from convnet import NN_ws, NN, ConvNet
from resnet import ResNet, ResNet_ws
from train import train

DIGITS = torch.arange(0, 10)
if __name__ == '__main__':

    if torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'


    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', default=42, type=int)
    parser.add_argument('-nepochs', default=50, type=int)
    parser.add_argument('-model', default=False, type=str)
    parser.add_argument('-test', default=False, action='store_true')
    parser.add_argument('-single', default=False, action='store_true')
    parser.add_argument('-no-weight-share', dest="no_weight_share", default=False, action='store_true')
    parser.add_argument('-no-aux-loss', dest="no_aux_loss", default=False, action='store_true')
    parser.add_argument('-baseline', dest="baseline", default=False, action='store_true')

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

        model_fn = f"results/models/nn-seed{seed}-al{int(not args.no_aux_loss)}-ws{int(not args.no_weight_share)}.pt"
        if (not os.path.isfile(model_fn) or not args.test):
            torch.manual_seed(seed)
            
            if args.model == "convnet":
                if args.no_weight_share:
                    model = NN().to(device)
                else:
                    model = NN_ws().to(device)   
            elif args.model == "resnet":
                if args.no_weight_share:
                    model = ResNet().to(device)
                else:
                    model = ResNet_ws().to(device)
            else:
                return ValueError("The model: -model argument has to be convnet or resnet")

            train_loss, val_loss, train_acc, val_acc = train(model, train_input, train_target, train_classes,
                    n_epochs, batch_size, device, validation_fraction=0.5, learning_rate=1e-3, use_aux_loss=not args.no_aux_loss)

            
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
            fig.savefig(f"results/plots/learning_curve_seed{seed}-al{int(not args.no_aux_loss)}-ws{int(not args.no_weight_share)}.png", dpi=200)
        
        else:
            if args.model == "convnet":
                if args.no_weight_share:
                    model = NN().to(device)
                else:
                    model = NN_ws().to(device)   
            elif args.model == "resnet":
                if args.no_weight_share:
                    model = ResNet().to(device)
                else:
                    model = ResNet_ws().to(device)
            else:
                return ValueError("The model: -model argument has to be convnet or resnet")

            model.load_state_dict(torch.load(model_fn))
            
        model.eval()

        out_1, out_2, out = model(test_input)
        out_classes = torch.argmax(out.to('cpu'), axis=1).to(int)
        class_1 = torch.argmax(out_1, axis=1)
        class_2 = torch.argmax(out_2, axis=1)

        accuracy = (out_classes == test_target).to(float).mean()
        print(f"Test accuracy = {accuracy}.")
        

        nrows = 5
        indices = torch.randperm(low=0, high=test_input.shape[0], size = (nrows,))

        fig, ax = plt.subplots(2, nrows, figsize=((2*nrows, 2*2)))

        for i in range(nrows):
            id_ = indices[i]
            print(id_)
            ax[0, i].imshow(test_input[id_][0, :, :])
            ax[1, i].imshow(test_input[id_][1, :, :])
            
            ax[0, i].set_title(f"{class_1[id_]} <= {class_2[id_]}?")
            ax[1, i].set_xlabel(f"Got {out_classes[id_]}. Expected {test_target[id_]}")

        fig.savefig(f"results/plots/test-seed{seed}-al{int(not args.no_aux_loss)}-ws{int(not args.no_weight_share)}.png", dpi=200)
    

    else:
        seeds = range(15)
        test_accuracies = []
        test_comp_accuracies = []
        test_class_accuracies = []
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        for seed in seeds:
            model_fn = f"results/models/nn-seed{seed}-al{int(not args.no_aux_loss)}-ws{int(not args.no_weight_share)}.pt"
            
            if (not os.path.isfile(model_fn) or not args.test):
            
                torch.manual_seed(seed)
                if args.no_weight_share:
                    model = NN().to(device)
                else:
                    model = NN_ws().to(device)   
            
                train_loss, val_loss, train_acc, val_acc = train(model, train_input, train_target, train_classes,
                        n_epochs, batch_size, device, validation_fraction=0.5, learning_rate=1e-3, use_aux_loss=not args.no_aux_loss)

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
                fig.savefig(f"results/plots/learning_curve_seed{seed}-al{int(not args.no_aux_loss)}-ws{int(not args.no_weight_share)}.png", dpi=200)

            else:
                if args.no_weight_share:
                    model = NN()
                else:
                    model = NN_ws()
                model.load_state_dict(torch.load(model_fn))



            model.eval()

            #indices = torch.randperm(test_input.shape[0])
            indices = torch.arange(test_input.shape[0])
            out_1, out_2, out = model(test_input[indices[:1000]])

            out_classes = torch.argmax(out.to('cpu'), axis=1)
            
            class_1 = torch.argmax(out_1, axis=1)
            class_2 = torch.argmax(out_2, axis=1)
            accuracy_1 = (class_1 == test_classes[indices[:1000], 0]).to(float).mean()
            accuracy_2 = (class_2 == test_classes[indices[:1000], 1]).to(float).mean()
            out_class_comp = (DIGITS[class_1] <= DIGITS[class_2]).to(int)
            accuracy = (out_classes == test_target[indices[:1000]]).to(float).mean()
            accuracy_comp = (out_class_comp == test_target[indices[:1000]]).to(float).mean()

            test_accuracies.append(accuracy)
            test_comp_accuracies.append(accuracy_comp)
            test_class_accuracies.append(accuracy_1)
            test_class_accuracies.append(accuracy_2)
            if not args.test:
                print(f"Test accuracy = {accuracy}.")
                print(f"Test accuracy comparison = {accuracy_comp}.")
                print(f"Test class accuracies = {accuracy_1}, {accuracy_2}.")

        test_accuracies = torch.Tensor(test_accuracies)
        test_comp_accuracies = torch.Tensor(test_comp_accuracies)
        test_class_accuracies = torch.Tensor(test_class_accuracies)
        print(f"Average test accuracy = {torch.mean(test_accuracies):.3f} +/- {torch.std(test_accuracies):.3f}")
        print(f"Average test comparison accuracy = {torch.mean(test_comp_accuracies):.3f} +/- {torch.std(test_comp_accuracies):.3f}")
        print(f"Average test class accuracies = {torch.mean(test_class_accuracies):.3f} +/- {torch.std(test_class_accuracies):.3f}")

        if args.test:
            train_losses = torch.load(f"results/data/ensemble_train_loss-al{int(not args.no_aux_loss)}-ws{int(not args.no_weight_share)}.pkl")
            val_losses = torch.load(f"results/data/ensemble_val_loss-al{int(not args.no_aux_loss)}-ws{int(not args.no_weight_share)}.pkl")
            train_accs = torch.load(f"results/data/ensemble_train_acc-al{int(not args.no_aux_loss)}-ws{int(not args.no_weight_share)}.pkl")
            val_accs = torch.load(f"results/data/ensemble_val_acc-al{int(not args.no_aux_loss)}-ws{int(not args.no_weight_share)}.pkl")
        else:
            train_losses = torch.stack(train_losses)
            val_losses = torch.stack(val_losses)
            train_accs = torch.stack(train_accs)
            val_accs = torch.stack(val_accs)
            torch.save(train_losses, f"results/data/ensemble_train_loss-al{int(not args.no_aux_loss)}-ws{int(not args.no_weight_share)}.pkl")
            torch.save(val_losses, f"results/data/ensemble_val_loss-al{int(not args.no_aux_loss)}-ws{int(not args.no_weight_share)}.pkl")
            torch.save(train_accs, f"results/data/ensemble_train_acc-al{int(not args.no_aux_loss)}-ws{int(not args.no_weight_share)}.pkl")
            torch.save(val_accs, f"results/data/ensemble_val_acc-al{int(not args.no_aux_loss)}-ws{int(not args.no_weight_share)}.pkl")
            
        
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
        fig.savefig(f"results/plots/learning_curve_ensemble-al{int(not args.no_aux_loss)}-ws{int(not args.no_weight_share)}.png", dpi=200)

          

            
