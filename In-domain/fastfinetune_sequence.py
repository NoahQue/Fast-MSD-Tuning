import argparse
import copy

from loader import MoleculeDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GlobalAttention, Set2Set
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split
import pandas as pd

import os
import shutil
import time
from sklearn.linear_model import LogisticRegression
import tempfile

from tensorboardX import SummaryWriter
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

criterion = nn.BCEWithLogitsLoss(reduction = "none")

def train(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) #y_true.shape[1]

def run(args, save_path):
    # Training settings


    args.use_early_stopping = args.dataset in ("muv", "hiv")
    args.scheduler = args.dataset in ("bace")
    
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    dataset = MoleculeDataset("./dataset/" + args.dataset, dataset=args.dataset)

    print(dataset)
    
    if args.split == "scaffold":
        smiles_list = pd.read_csv('./dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    else:
        raise ValueError("Invalid split option.")

    print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    #set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    if not args.input_model_file == "":
        print("load pretrained model from:", args.input_model_file)
        model.from_pretrained(args.input_model_file)
    
    model.to(device)

    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.3)
    else:
        scheduler = None

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    best_val_acc = 0
    final_test_acc = 0
    best_model = None

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train(args, model, device, train_loader, optimizer)
        if scheduler is not None:
            scheduler.step()

        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(args, model, device, val_loader)
        test_acc = eval(args, model, device, test_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            final_test_acc = test_acc
            best_model = copy.deepcopy(model.state_dict())

        print("train: %f val: %f test: %f" %(train_acc, val_acc, final_test_acc))

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)


    print('Best epoch:', val_acc_list.index(max(val_acc_list)) + 1)
    print('Best auc: ', test_acc_list[val_acc_list.index(max(val_acc_list))])

    # Save the best model
    torch.save(best_model, save_path)
    best_test = test_acc_list[val_acc_list.index(max(val_acc_list))]

    return best_test

def only_train_task_head(args, save_path):
    args.use_early_stopping = args.dataset in ("muv", "hiv")
    args.scheduler = args.dataset in ("bace")
    
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # Determine number of tasks based on dataset
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    # Set up dataset
    dataset = MoleculeDataset("./dataset/" + args.dataset, dataset=args.dataset)

    print(dataset)
    
    if args.split == "scaffold":
        smiles_list = pd.read_csv('./dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    else:
        raise ValueError("Invalid split option.")

    print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    # Set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK=args.JK, drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)
    if not args.input_model_file == "":
        print("load pretrained model from:", args.input_model_file)
        model.from_pretrained(args.input_model_file)
    
    model.to(device)

    # Freeze GNN parameters
    for param in model.gnn.parameters():
        param.requires_grad = False
    
    # Unfreeze graph_pred_linear and pool parameters
    for param in model.graph_pred_linear.parameters():
        param.requires_grad = True
    if isinstance(model.pool, GlobalAttention) or isinstance(model.pool, Set2Set):
        for param in model.pool.parameters():
            param.requires_grad = True

    # Set up optimizer (only for graph_pred_linear and pool layers)
    model_param_group = []
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr * args.lr_scale})
    if isinstance(model.pool, GlobalAttention) or isinstance(model.pool, Set2Set):
        model_param_group.append({"params": model.pool.parameters(), "lr": args.lr * args.lr_scale})

    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.3)
    else:
        scheduler = None

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    best_val_acc = 0
    final_test_acc = 0
    best_model = None

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))
        
        # Train for one epoch
        train(args, model, device, train_loader, optimizer)
        if scheduler is not None:
            scheduler.step()

        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(args, model, device, val_loader)
        test_acc = eval(args, model, device, test_loader)

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            final_test_acc = test_acc
            best_model = copy.deepcopy(model.state_dict())

        print("train: %f val: %f test: %f" % (train_acc, val_acc, final_test_acc))

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)

    # Print best results
    print('Best epoch:', val_acc_list.index(max(val_acc_list)) + 1)
    print('Best auc: ', test_acc_list[val_acc_list.index(max(val_acc_list))])

    # Save the best model
    torch.save(best_model, save_path)
    best_test = test_acc_list[val_acc_list.index(max(val_acc_list))]

    return best_test

def fast_train(args, save_path):
    # Training settings
    args.create_projection = True
    
    args.use_early_stopping = args.dataset in ("muv", "hiv")
    args.scheduler = args.dataset in ("bace")
    
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
        args.epochs = 20
        args.project_dim = 1000 
        
    elif args.dataset == "hiv":
        num_tasks = 1
        args.epochs = 5
        args.project_dim = 1000 
        
    elif args.dataset == "pcba":
        num_tasks = 128
        args.epochs = 1
        args.project_dim = 500  
    
    elif args.dataset == "muv":
        num_tasks = 17
        args.epochs = 5
        args.project_dim = 1000
    
    elif args.dataset == "bace":
        num_tasks = 1
        args.epochs = 10
        args.project_dim = 500
        
    elif args.dataset == "bbbp":
        num_tasks = 1
        args.epochs = 5
        args.project_dim = 500
    
    elif args.dataset == "toxcast":
        num_tasks = 617
        args.epochs = 20
        args.project_dim = 1000 
    
    elif args.dataset == "sider":
        num_tasks = 27
        args.epochs = 5
        args.project_dim = 1000
    
    elif args.dataset == "clintox":
        num_tasks = 2
        args.epochs = 30
        args.project_dim = 2000
        
    else:
        raise ValueError("Invalid dataset name.")

    # Set up dataset
    dataset = MoleculeDataset("./dataset/" + args.dataset, dataset=args.dataset)

    print(dataset)
    
    if args.split == "scaffold":
        smiles_list = pd.read_csv('./dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    else:
        raise ValueError("Invalid split option.")

    print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    # Set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    if not args.input_model_file == "":
        print("Load pretrained model from:", args.input_model_file)
        model.from_pretrained(args.input_model_file)
    
    model.to(device)

    # Set up optimizer
    # Different learning rate for different parts of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.3)
    else:
        scheduler = None

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    best_val_acc = 0
    final_test_acc = 0
    best_model = None

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train(args, model, device, train_loader, optimizer)
        if scheduler is not None:
            scheduler.step()

        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader)
        else:
            print("Omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(args, model, device, val_loader)
        test_acc = eval(args, model, device, test_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            final_test_acc = test_acc
            best_model = copy.deepcopy(model.state_dict())

        print("train: %f val: %f test: %f" % (train_acc, val_acc, final_test_acc))

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)

    print('Best epoch:', val_acc_list.index(max(val_acc_list)) + 1)
    print('Best auc: ', test_acc_list[val_acc_list.index(max(val_acc_list))])
    # Save the best model
    # torch.save(best_model, save_path)
    best_test = test_acc_list[val_acc_list.index(max(val_acc_list))]
    if args.dataset == "pcba" or args.dataset == "toxcast":
        torch.save(best_model, save_path)
        return test_acc
        
    # Perform gradient calculation
    # Determine the total model parameter dimension (for random projection)
    gradient_dim = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            gradient_dim += param.numel()

    # Set up directory to save projection matrix
    gradients_dir = f"./fast_test/gradients/{args.dataset}_{args.project_dim}_{num_tasks}_epoch{args.epochs}"

    # Create directory if it doesn't exist
    if not os.path.exists(gradients_dir):
        os.makedirs(gradients_dir)

    # Check if projection matrix file already exists
    projection_matrix_path = f"{gradients_dir}/projection_matrix_{args.run}.npy"
    if not os.path.exists(projection_matrix_path):
        # If not, create a new projection matrix
        project_dim = args.project_dim
        matrix_P = (2 * np.random.randint(2, size=(gradient_dim, project_dim)) - 1).astype(np.float32)
        matrix_P *= 1 / np.sqrt(project_dim)
        # Save the projection matrix
        np.save(projection_matrix_path, matrix_P)
        print(f"Projection matrix created and saved at {projection_matrix_path}.")
    else:
        # If exists, load the projection matrix from the file
        matrix_P = np.load(projection_matrix_path)
        print(f"Projection matrix loaded from {projection_matrix_path}.")

    # Gradient calculation
    gradients_list = []
    outputs_list = []
    labels_list = []

    for task_idx in range(num_tasks):
        print(f"Processing task {task_idx}...")

        # Define file paths to save gradients, outputs, and labels
        gradients_path = f"{gradients_dir}/task_{task_idx}_gradients.npy"
        outputs_path = f"{gradients_dir}/task_{task_idx}_outputs.npy"
        labels_path = f"{gradients_dir}/task_{task_idx}_labels.npy"

        # Check if gradients, outputs, and labels files already exist
        if not os.path.exists(gradients_path) or not os.path.exists(outputs_path) or not os.path.exists(labels_path):
 
            gradients, outputs, labels = get_task_gradients_optimized(
                model=model,
                train_loader=train_loader,
                task_idx=task_idx,
                device=device,
                projection=matrix_P,
                save_dir=f"{gradients_dir}/tmp"
            )
            
            # Save gradients, outputs, and labels
            print(f"Saving gradients for task {task_idx}, shape: {gradients.shape}")
            np.save(gradients_path, gradients)
            np.save(outputs_path, outputs)
            np.save(labels_path, labels)
        else:
            print(f"Gradients, outputs, and labels for task {task_idx} already exist. Skipping computation.")
 
    # Logistic regression approximation
    task_idxes = range(num_tasks)  
    print("Selected task idxes", task_idxes)

    task_to_gradients = {}
    task_to_outputs = {}
    task_to_labels = {}
    for task_idx in task_idxes:
        gradients = np.load(f"{gradients_dir}/task_{task_idx}_gradients.npy")[:, :gradient_dim]
        outputs = np.load(f"{gradients_dir}/task_{task_idx}_outputs.npy")
        labels = np.load(f"{gradients_dir}/task_{task_idx}_labels.npy")
        task_to_gradients[task_idx] = gradients
        task_to_outputs[task_idx] = outputs
        task_to_labels[task_idx] = labels

    # Combine gradients and labels from all tasks
    gradients = np.concatenate([task_to_gradients[task_idx] for task_idx in task_idxes], axis=0)
    labels = np.concatenate([task_to_labels[task_idx] for task_idx in task_idxes], axis=0)

    # Split into training and testing sets
    train_num = int(len(gradients) * 0.8)
    train_gradients, train_labels = gradients[:train_num], labels[:train_num]
    test_gradients, test_labels = gradients[train_num:], labels[train_num:]

    # Use Logistic Regression to fit gradient-label relationship
    clf = LogisticRegression(random_state=0, penalty='l2', C=1e-2)
    clf.fit(train_gradients, train_labels)
    print(f"Test Accuracy: {clf.score(test_gradients, test_labels)}")

    # Project Logistic Regression coefficients back into model parameter space
    proj_coef = clf.coef_.copy().flatten().reshape(-1, 1)
    print("Projected Coef L2 Norm:", np.linalg.norm(proj_coef))
    coef = matrix_P @ proj_coef.flatten()
    print("Reconstructed Coef L2 Norm:", np.linalg.norm(coef))

    # Generate new model parameters
    new_state_dict = generate_state_dict(model, model.state_dict(), coef, device)
    model.load_state_dict(new_state_dict, strict=False)

    torch.save(model.state_dict(), save_path)
    
    test_acc = eval(args, model, device, test_loader)
    print('Few epoch Best auc: ', test_acc_list[val_acc_list.index(max(val_acc_list))])
    print("Fast test_acc:", test_acc)
    
    return test_acc



def generate_state_dict(model, state_dict, coef, device):
    # reshape coef
    new_state_dict = {}; cur_len = 0
    for key, param in model.named_parameters():
        param_len = np.prod(param.shape)
        if "project" in key: 
            new_state_dict[key] = state_dict[key].clone()
            continue
        new_state_dict[key] = state_dict[key].clone() + \
            torch.FloatTensor(coef[cur_len:cur_len+param_len].reshape(param.shape)).to(device)
        cur_len += param_len
    return new_state_dict


def get_task_gradients(model, train_loader, task_idx, device, projection=None, max_gradients_in_gpu=1000):

    model.train()  # Set model to training mode
    all_gradients = []  # Store gradients for all samples (will eventually be moved to CPU)
    model_outputs = []  # Store model outputs for all samples
    labels = []  # Store true labels for all samples

    # Temporarily store gradients on GPU
    gpu_gradients = []

    for batch in train_loader:
        # Get input features, edges, edge attributes, batch indices, etc.
        data = batch.to(device)
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Forward pass, compute task-specific output
        outputs = model(x, edge_index, edge_attr, batch)
        y = data.y.view(outputs.shape)
        # task_mask = (y[:, task_idx] != -1)  # Avoid invalid labels (-1)
        task_mask = y[:, task_idx]**2 > 0
        # Get valid outputs for the current task
        task_outputs = outputs[task_mask, task_idx]
        model_outputs.append(task_outputs.detach().to("cpu").numpy())  # Save model outputs
        labels.append(y[task_mask, task_idx].detach().to("cpu").numpy())  # Save true labels

        # Calculate gradients for each sample
        for i in range(len(task_outputs)):
            # Calculate loss for each individual sample
            loss = task_outputs[i]
            # Compute gradients for the current sample with respect to model parameters
            tmp_gradients = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=False)
            # Flatten each parameter gradient and concatenate them
            tmp_gradients = torch.cat([g.view(-1) for g in tmp_gradients]).to(device)
            gpu_gradients.append(tmp_gradients)  # Temporarily store on specified GPU

            # If the number of gradients stored on GPU exceeds the limit, transfer to CPU
            if len(gpu_gradients) >= max_gradients_in_gpu:
                # Transfer gradients from GPU to CPU and store them
                gpu_gradients = torch.stack(gpu_gradients).to("cpu").numpy()
                all_gradients.append(gpu_gradients)
                gpu_gradients = []  # Clear GPU gradient cache
                torch.cuda.empty_cache()  # Clean up GPU memory

    # Process any remaining GPU gradients
    if len(gpu_gradients) > 0:
        gpu_gradients = torch.stack(gpu_gradients).to("cpu").numpy()
        all_gradients.append(gpu_gradients)
        gpu_gradients = []
        torch.cuda.empty_cache()  # Clean up GPU memory

    # Combine all gradients
    all_gradients = np.concatenate(all_gradients, axis=0)

    # If a proj



def get_task_gradients_optimized(model, train_loader, task_idx, device, projection=None, max_gradients_in_gpu=1000, save_dir="./gradients"):
    """
    Calculate the gradients for a single task and save the gradients to a specified directory to save memory.
    """
    # Create directory to save gradients
    os.makedirs(save_dir, exist_ok=True)

    model.to(device)  # Ensure the model is on the target device
    model.train()
    model_outputs = []
    labels = []
    gradient_files = []  # Store paths of gradient files

    # Preload the projection matrix to GPU
    if projection is not None:
        projection_tensor = torch.tensor(projection, device=device, dtype=torch.float32)
    else:
        projection_tensor = None

    for batch_idx, batch in enumerate(train_loader):
        data = batch.to(device, non_blocking=True)  # Ensure data is transferred to GPU
        x, edge_index, edge_attr, batch = data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device), data.batch.to(device)
        
        # Forward pass
        outputs = model(x, edge_index, edge_attr, batch)  # Execute on GPU
        y = data.y.view(outputs.shape)
        task_mask = y[:, task_idx]**2 > 0
        task_outputs = outputs[task_mask, task_idx]
        
        # Check if task_outputs is empty
        if task_outputs.numel() == 0:
            print(f"Warning: No valid task outputs for batch {batch_idx}. Skipping this batch.")
            continue
        
        model_outputs.append(task_outputs.detach().cpu().numpy())
        labels.append(y[task_mask, task_idx].detach().cpu().numpy())

        batch_gradients = []  # Store gradients for the current batch

        # Calculate gradients for each sample
        for sample_idx in range(len(task_outputs)):
            loss = task_outputs[sample_idx]
            tmp_gradients = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=False)
            tmp_gradients = torch.cat([g.view(-1) for g in tmp_gradients])

            # If a projection matrix exists, reduce dimensionality (apply projection on GPU)
            if projection_tensor is not None:
                tmp_gradients = torch.matmul(tmp_gradients, projection_tensor)

            batch_gradients.append(tmp_gradients.cpu().numpy())

        # Check if batch_gradients is empty
        if len(batch_gradients) == 0:
            print(f"Warning: No gradients for batch {batch_idx}. Skipping this batch.")
            continue  # Skip the current batch
        
        # Save the gradients of the current batch to file
        batch_gradients = np.stack(batch_gradients, axis=0)
        gradient_file = os.path.join(save_dir, f"gradient_task{task_idx}_batch{batch_idx}.npy")
        np.save(gradient_file, batch_gradients)
        gradient_files.append(gradient_file)

        # Clean up intermediate variables and release memory
        del data, x, edge_index, edge_attr, batch, task_outputs, batch_gradients
        torch.cuda.empty_cache()

    # Load and combine all gradients from disk
    all_gradients = np.concatenate([np.load(file) for file in gradient_files], axis=0)

    # Combine model outputs and labels
    model_outputs = np.concatenate(model_outputs)
    labels = np.concatenate(labels)

    return all_gradients, model_outputs, labels



def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=4,
                        help='which GPU to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--scheduler', action="store_true", default=False)
    
    '''Projection'''
    parser.add_argument("--create_projection", action="store_true")
    parser.add_argument("--project_dim", type=int, default=2000)
    parser.add_argument("--run", type=int, default=0)
    
    args = parser.parse_args()
    
    reverse_task_index = {0: 'bace', 1: 'bbbp', 2: 'tox21', 3: 'sider', 4: 'hiv', 5: 'muv', 6: 'pcba', 7: 'clintox', 8: 'toxcast'}
    
    # Record start time
    start_time = time.time()

    # Read sampled_sequences.csv file
    df = pd.read_csv('./gain/sampled_sequences_all_len4.csv')
  
    df['Task Sequence'] = df['Task Sequence'].apply(lambda x: list(map(int, x.strip('"').split(','))))
    sequences_list = df['Task Sequence']
    sequences_list = sequences_list
    
    # If the results directory doesn't exist, create it
    if not os.path.exists("./results/"):
        os.makedirs("./results/")
      
    # Define CSV file path
    result_file = './gain/result.csv'
    result_dir = os.path.dirname(result_file)

    # Ensure the directory exists, if not create it
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # If the CSV file doesn't exist, create it and write the header
    if not os.path.exists(result_file):
        # Use pandas to create an empty DataFrame and save it to the CSV file
        df = pd.DataFrame(columns=['Task Sequence', 'Best Test Result'])
        df.to_csv(result_file, index=False)
        
    for task_seq in sequences_list:
        print(task_seq)
        task_name_sequence = [reverse_task_index[index] for index in task_seq]
        # print(task_name_sequence)
        for i, dataset in enumerate(task_name_sequence):
            args.dataset = dataset
            task_seq_name = ''
            for k in range(i+1):
                task_seq_name += str(task_seq[k])
            # print("task_seq_name:", task_seq_name)
            save_model_file = f"./results/{task_seq_name}.pth"     
            if os.path.exists(save_model_file):
                continue
            else:
                if i == 0:
                    args.input_model_file = "./pretrain_model/contextpred.pth"
                else:
                    last_model_name = task_seq_name[0:-1]
                    print("last_model_name:", last_model_name)
                    args.input_model_file = f"./results/{last_model_name}.pth" 
                best_test = fast_train(args, save_model_file)
        

                # Use pandas to write each experimental result to the CSV file (incremental saving)
                result_data = {'Task Sequence': task_seq_name, 'Best Test Result': best_test}
                result_df = pd.DataFrame([result_data])  # Create a single-row DataFrame
                result_df.to_csv(result_file, mode='a', header=False, index=False)  # Incrementally write to CSV file

    # Record end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the time taken in seconds
    print(f"Time taken for the process: {elapsed_time:.2f} seconds")

    # If you need to print the elapsed time in minutes and seconds, use the following code:
    minutes = elapsed_time // 60
    seconds = elapsed_time % 60
    print(f"Time taken for the process: {int(minutes)} minutes and {seconds:.2f} seconds")

if __name__ == "__main__":
    main()
