import tqdm
import torch

from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, accuracy_score
def eval_metrics(actual, pred):
    return f1_score(pred,actual), accuracy_score(pred,actual),recall_score(pred,actual),precision_score(pred,actual)

def training(model,optimizer,loss_function,train_loader,val_loader,NUMBER_OF_EPOCHS,device,scheduler):


    all_train_losses = []
    all_train_accuracy = []
    all_val_losses = []
    all_val_accuracy = []
    for epoch in range(NUMBER_OF_EPOCHS):
        # training
        epoch_loss = 0
        correct = 0
        acc_list = []
        epoch_all = 0
        model.train()
        with tqdm.tqdm(enumerate(train_loader), total=len(train_loader)) as pbar:
            for i, (X, y) in pbar:
                X , y = X.to(device), y.to(device,dtype = torch.long)
                optimizer.zero_grad()
                outputs = model(X)
                loss = loss_function (outputs , y)
                loss.backward()
                optimizer.step()
                epoch_loss += outputs.shape[0] * loss.item()
                epoch_all += y.size(0)
                predicted = torch.max(outputs.data,1)[1]
                correct += (predicted == y).sum().item()
                pbar.set_description(f'epoch {epoch } - train Loss: {epoch_loss / (i + 1):.3e} - train Acc: {correct * 100. / epoch_all:.2f}%')
        loss_list = []
        scheduler.step()
        model.eval() 
        with torch.no_grad():
            main = []
            pre = []
            epoch_loss = 0
            epoch_all = 0
            correct = 0
            corr = 0
            tot = 0
            with tqdm.tqdm(enumerate(val_loader), total=len(val_loader)) as pbar:
                for i, (X, y) in pbar:
                    X , y = X.to(device), y.to(device, dtype = torch.long)
                    out = model(X)
                    los = loss_function (out , y).item()
                    main.append(y)
                    epoch_loss += los
                    loss_list.append(los)
                    predicts = torch.max(out.data,1)[1]
                    pre.append(predicts)
                    epoch_all+=y.size(0)
                    tot += y.size(0)
                    correct += (predicts == y).sum().item()
                    pbar.set_description(f'epoch {epoch } - val Loss: {epoch_loss / (i + 1):.3e} - val Acc: {correct * 100. / epoch_all:.2f}%')
               
