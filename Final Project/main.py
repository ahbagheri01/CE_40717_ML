import sklearn
import json
import torch
import mlflow
import mlflow.pytorch
import Preprocess as ps
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
import os
import warnings
import sys
import pandas as pd
import pickle
import numpy as np
import numpy as np
import torch
from urllib.parse import urlparse
import torch.nn.functional as F
from torch import nn
import pandas as pd
import DWModels as MD
import Preprocess as ps
import tqdm
import logging
from collections import Counter
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, accuracy_score
import ML_train as tr
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

state_dict_path = f'content/modelD.pt'
column_idx_path = f'content/column_idx.pkl'
embeddings_input_path = f'content/embeddings_input.pkl'
cont_cols_path = f'content/cont_cols.pkl'
model_path = f"content/Recommendation"
artifacts = {
    "state_dict_model": state_dict_path,
    "column_idx": column_idx_path,
    "embeddings_input": embeddings_input_path,
    "cont_cols":cont_cols_path
}
conda_env = {
    'channels': ['defaults'],
    'dependencies': [
            f'python=3.9.7',
      {
          'pip':[
            f'mlflow=={mlflow.__version__}',
            f'torch=={torch.__version__}',
            f'numpy=={np.__version__}',
            f'sklearn == {sklearn.__version__}'
    ]
      }
    ],
    'name': 'Recommendation'
}
class ModelWrapper(mlflow.pyfunc.PythonModel):

  def load_context(self, context):
    import DWModels as MD
    import Preprocess as ps
    import torch
    self._p = ps.preprocess()

    # Load in and deserialize the embeddings
    print(context.artifacts)
    with open(context.artifacts["column_idx"], 'rb') as handle:
      self._column_idx = pickle.load(handle)
    
    # load in and deserialize the model tokenizer
    with open(context.artifacts["embeddings_input"], 'rb') as handle:
      self._embeddings_input = pickle.load(handle)
    
    with open(context.artifacts["cont_cols"], 'rb') as handle:
      self._cont_cols = pickle.load(handle)

    model = MD.TabMlp(
    mlp_hidden_dims=[500,400,300,200, 100, 100],
    column_idx=self._column_idx,
    embed_input=self._embeddings_input,
    mlp_dropout=[0.2,0.3,0.2,0.2,0.2,0.2],
    continuous_cols=self._cont_cols,
    mlp_batchnorm=True,
    pred_dim = 2)
    
    self._model = model
    self._model.load_state_dict(torch.load(context.artifacts["state_dict_model"]))
    self._model.eval()
    
  def predict(self, context, input_model):
    input_m = torch.Tensor(self._p.prepro_test(input_model))
    output = self._model(input_m)
    predicted = torch.max(output.data,1)[1]
    return predicted.numpy()

def startModel():
    f = open('model_run.txt', mode='rt')
    run_id = f.read()
    model_name = "model"
    if run_id == "":
        name = f'Recommendation Model'
        try:
            experiment_id = mlflow.get_experiment_by_name(name).experiment_id
        except:
            experiment_id = mlflow.create_experiment(name)
        mlflow.set_experiment(experiment_id=experiment_id)

        warnings.filterwarnings("ignore")
        data = pd.read_csv('PreModule/datasets/train_dataset.csv')
        p = ps.preprocess()

        y,X,column_idx,embeddings_input,cont_cols = p.prepro_train(data)

        mlp_hidden_dims=[500,400,300,200, 100, 100]
        column_idx=column_idx
        embed_input=embeddings_input
        mlp_dropout=[0.2,0.3,0.2,0.2,0.2,0.2]
        continuous_cols=cont_cols
        mlp_batchnorm=True
        pred_dim = 2
        model = MD.TabMlp(
        mlp_hidden_dims=mlp_hidden_dims,
        column_idx=column_idx,
        embed_input=embeddings_input,
        mlp_dropout=mlp_dropout,
        continuous_cols=continuous_cols,
        mlp_batchnorm=mlp_batchnorm,
        pred_dim = pred_dim)
        X_train,X_test,y_train,y_test= train_test_split(X,
                                                        y,
                                                        test_size=0.25,
                                                        random_state=42,
                                                        shuffle=True)
    
        count=Counter(y_train)

        class_count=np.array([count[0],count[1]])

        weight=1./class_count
        samples_weight = np.array([weight[int(t)] for t in y_train])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
        train_subset = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        val_subset = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
        loss_function = nn.CrossEntropyLoss()
        BATCH_SIZE = 256

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        model.to(device);
        lr = 0.01
        step_size = 10 
        gamma=0.1

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        NUMBER_OF_EPOCHS = 22
        train_loader = DataLoader(dataset=train_subset, batch_size=BATCH_SIZE, sampler=sampler)
        val_loader = DataLoader(dataset=val_subset, shuffle=False, batch_size=BATCH_SIZE)

        with mlflow.start_run() as run:
            tr.training(model,optimizer,loss_function,train_loader,val_loader,NUMBER_OF_EPOCHS,device,scheduler)
            model.eval()
            xt = torch.Tensor(X_test)
            xt = xt.to(device)
            t = model(xt)
            predicted = torch.max(t.data,1)[1]
            (f1, acc, recall,precision) = tr.eval_metrics(y_test, predicted.cpu())
            model.eval()
            xt = torch.Tensor(X_train)
            xt = xt.to(device)
            t = model(xt)
            predicted = torch.max(t.data,1)[1]
            (f1t, acct, recallt,precisiont) = tr.eval_metrics(y_train, predicted.cpu())
            parametrs = {
                "NUMBER_OF_EPOCHS":NUMBER_OF_EPOCHS,
                "loss_function": "nn.CrossEntropyLoss",
                "optimizer":"Adam",
                "lr":lr,
                "step_size":step_size,
                "gamma":gamma,
                "BATCH_SIZE":BATCH_SIZE,
                "mlp_hidden_dims":mlp_hidden_dims,
                "column_idx":column_idx,
                "embeddings_input":embeddings_input,
                "mlp_dropout":mlp_dropout,
                "continuous_cols":continuous_cols,
                "mlp_batchnorm":mlp_batchnorm,
                "pred_dim":pred_dim,  
            }
            mlflow.log_param("parametrs", parametrs)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("acc", acc)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("f1t", f1t)
            mlflow.log_metric("acct", acct)
            mlflow.log_metric("recallt", recallt)
            mlflow.log_metric("precisiont", precisiont)
            print(f'parametrs : {parametrs}\nf1 : {f1}\nacc : {acc}\nrecall {recall}\nprecision : {precision}\
            \nf1t : {f1t}\nacct : {acct}\nrecallt {recallt}\nprecisiont : {precisiont}')

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            # Model registry does not work with file store
            #os.system(f"rm -r {model_path}")
            torch.save(model.state_dict(), state_dict_path)
            with open(column_idx_path, 'wb') as handle:
                pickle.dump(column_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(embeddings_input_path, 'wb') as handle:
                pickle.dump(embeddings_input, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            with open(cont_cols_path, 'wb') as handle:
                pickle.dump(cont_cols, handle, protocol=pickle.HIGHEST_PROTOCOL)
            python_model=ModelWrapper()
            mlflow.pyfunc.log_model(artifact_path =model_path,python_model=python_model,
                         artifacts=artifacts,
                         conda_env=conda_env)
            
            
            
            
            if tracking_url_type_store != "file":
                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.pytorch.log_model(model, "model", model_name)
            else:
                mlflow.pytorch.log_model(model, "model")
            f = open('model_run.txt', mode='wt')
            f.write(str(run.info.run_id))
            f.close()

            os.system(f'mlflow models serve -m "./mlruns/4/{run.info.run_id}/artifacts/content/Recommendation" --no-conda -h 0.0.0.0 -p 8080')
    else:
        os.system(f'mlflow models serve -m "./mlruns/4/{run_id}/artifacts/content/Recommendation" --no-conda -h 0.0.0.0 -p 8080')
if __name__ == '__main__':
    startModel()          
