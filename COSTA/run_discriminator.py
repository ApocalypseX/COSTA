import random
import os
import torch
import datetime
import  json
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from torch.backends import cudnn
from buffer.OfflineSafeBuffer import *
from torch.utils.data import DataLoader, TensorDataset
from nets import *
from train.discriminator_train import *
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
from modules import EnsembleDiscriminatorModel

ex = Experiment()

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False
    cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(seed)

def read_json_dict(path):
    with open(path, "r", encoding='utf-8') as r:
        dic = json.load(r)
    return dic

@ex.capture(prefix='discriminator_params')
def run(_config):
    seed_torch(_config["seed"])
    buffer=SimpleSafeReplayBuffer(_config["max_buffer_size"], _config["state_dim"], _config["action_dim"], _config["goal_radius"])
    buffer.init_buffer(_config["data_path"])
    
    data=buffer.sample_all()
    permute=np.arange(data["actions"].shape[0])
    random.shuffle(permute)
    train_size=int(data["actions"].shape[0]*_config["train_ratio"])
    trainset = TensorDataset(torch.tensor(data["actions"][permute][:train_size], dtype=torch.float32), torch.tensor(data["observations"][permute][:train_size], dtype=torch.float32), torch.tensor(data["next_observations"][permute][:train_size], dtype=torch.float32), torch.tensor(data["costs"][permute][:train_size], dtype=torch.float32))
    trainloader = DataLoader(trainset, batch_size=_config["batch_size"], shuffle=True)
    testset = TensorDataset(torch.tensor(data["actions"][permute][train_size:], dtype=torch.float32), torch.tensor(data["observations"][permute][train_size:], dtype=torch.float32), torch.tensor(data["next_observations"][permute][train_size:], dtype=torch.float32), torch.tensor(data["costs"][permute][train_size:], dtype=torch.float32))
    testloader = DataLoader(testset, batch_size=1, shuffle=False)
    
    #model=Discriminator(2*_config["state_dim"]+_config["action_dim"], _config["hidden_dims"], output_dim=_config["output_dim"],dropout_rate=_config["dropout_rate"])
    model = EnsembleDiscriminatorModel(_config["state_dim"],_config["action_dim"],_config["hidden_dims"],num_ensemble=5,num_elites=5,weight_decays=_config["weight_decay"])
    optimizer = torch.optim.Adam(model.parameters(), lr=_config["learning_rate"], eps=1e-5)
    loss = torch.nn.BCELoss(reduce=False)
    
    train(model, trainloader, testloader, buffer, loss, optimizer, _config["epoch_num"], _config["path"]+"/tb", _config["path"]+"/model", _config["eval_size"], _config["root_path"])

@ex.main
def my_main():
    run()

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='ant-dir-safe')
    parser.add_argument('--goal', type=int, default=1)
    args = parser.parse_args()

    config_dict=read_json_dict("configs/"+args.task+".json")
    mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')
    config_dict["discriminator_params"]["root_path"]=os.path.join(config_dict["discriminator_params"]["path"],args.task+"_"+str(args.goal),"discriminator")
    config_dict["discriminator_params"]["path"]=os.path.join(config_dict["discriminator_params"]["path"],args.task+"_"+str(args.goal),"discriminator",mkfile_time)
    config_dict["discriminator_params"]["data_path"]=os.path.join(config_dict["discriminator_params"]["data_path"],args.task+"_"+str(args.goal),"offline_buffer.npz")
    root_path=config_dict["discriminator_params"]["path"]
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    ex.add_config(config_dict)
    file_obs_path = os.path.join(root_path,"sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))
    ex.run()