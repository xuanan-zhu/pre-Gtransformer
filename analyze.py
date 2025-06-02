import os

import numpy as np
import pandas as pd
import torch
from matplotlib import patches
from matplotlib.lines import Line2D
import pickle
from sklearn.metrics import roc_auc_score
from visualizer import get_local
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from torch.utils.data import TensorDataset, DataLoader

get_local.activate()

from model.gwas_transformer_base_model import clsDNA, get_pos_embedding
from data_loader_tools import *

parser = argparse.ArgumentParser(description='base_model_vision')
parser.add_argument('--datasetX', type=int, default='0')
parser.add_argument('--fold', type=int, default='3')
data_name = 'adni_data'
args = parser.parse_args()


def get_feature_importance(average_attention):
    importance = np.mean(average_attention, axis=0) # Average across all features that attend to it
    return importance


def find_important_feature_combinations(average_attention, threshold=0.01):

    num_features = average_attention.shape[0]

    i_indices, j_indices = np.triu_indices(num_features, k=1)

    vals_ij = average_attention[i_indices, j_indices]
    vals_ji = average_attention[j_indices, i_indices]

    mask = (vals_ij > threshold) | (vals_ji > threshold)

    valid_i = i_indices[mask]
    valid_j = j_indices[mask]
    total_weights = vals_ij[mask] + vals_ji[mask]

    combinations = [((i, j), w) for i, j, w in zip(valid_i, valid_j, total_weights)]
    combinations.sort(key=lambda x: x[1], reverse=True)
    return combinations



def save_as_csv(significant_pairs, significant_nodes, snp_names, result_dir):

    pairs_df = pd.DataFrame(
        [(i, j, snp_names[i], snp_names[j], w) for (i, j), w in significant_pairs.items()],
        columns=['Feature_A', 'Feature_B', 'SNP_A', 'SNP_B', 'Weight']
    )
    pairs_csv_path = os.path.join(result_dir, 'significant_pairs.csv')
    pairs_df.to_csv(pairs_csv_path, index=False)
    print(f"Saved feature pairs to {pairs_csv_path} ({len(pairs_df)} rows)")


    nodes_df = pd.DataFrame(
        [(node, snp_names[node], count) for node, count in significant_nodes.items()],
        columns=['Feature_ID', 'SNP_Name', 'Interaction_Count']
    )
    nodes_csv_path = os.path.join(result_dir, 'significant_nodes.csv')
    nodes_df.to_csv(nodes_csv_path, index=False)
    print(f"Saved significant nodes to {nodes_csv_path} ({len(nodes_df)} rows)")

import numpy as np
from sklearn.cluster import KMeans


def extract_significant_features(final_avg_attn_per_layer_head):

    aggregated_matrix = np.sum(final_avg_attn_per_layer_head, axis=(0, 1))  


    mean_val = np.mean(aggregated_matrix)
    std_val = np.std(aggregated_matrix)
    threshold = mean_val + 3 * std_val


    num_features = aggregated_matrix.shape[0]
    i_indices, j_indices = np.triu_indices(num_features, k=1) 
    significant_pairs = {}
    significant_nodes = {}

    for i, j in zip(i_indices, j_indices):
        if aggregated_matrix[i, j] > threshold and aggregated_matrix[j, i] > threshold:

            key = tuple(sorted((i, j)))
            significant_pairs[key] = significant_pairs.get(key, 0) + (aggregated_matrix[i, j] + aggregated_matrix[j, i])/2


            significant_nodes[i] = significant_nodes.get(i, 0) + 1
            significant_nodes[j] = significant_nodes.get(j, 0) + 1

    return significant_pairs, significant_nodes, aggregated_matrix, threshold



model = clsDNA(
    vocab=40,
    seq_len=2277,
    num_classes=2,
    dim=int(128),
    depth=4,
    heads=4,
    mlp_dim=256,
    dropout=0.4,
    emb_dropout=0.4,
    get_last_feature=False,
    pool='mean',
    use_auto_pos=False,
    snp_count_list=None,
)
seed = 1
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
print("now using device:", device)

datasetX = args.datasetX
print("datasetX:", datasetX)



print("loading best auc model...")
checkpoint_path = './checkpoint/new_ADNI_model_tune/{}/new_fold{}/'.format(data_name,args.fold)
checkpoint_path = checkpoint_path + "clsDNA" + 'ckpt.t_best_model'
checkpoint = torch.load(checkpoint_path, map_location=device) 

model.load_state_dict(checkpoint['model'])



chr_pos = np.loadtxt('./data/{}/fold_{}/snp_content_test_snp.txt'.format(data_name, args.fold), delimiter=' ', dtype=str)
t_data = np.loadtxt('./data/{}/fold_{}/genetype_test_snp.txt'.format(data_name, args.fold), delimiter=' ', dtype=str)


test_x_data, test_y_data = data_xy_transform_to_tensor_all(t_data)
test_x_data = test_x_data.to(torch.float32)


pos_data0 = data_pos_transform_to_tensor_all(chr_pos)
pos_data0 = pos_data0.to(device)
pos_data0 = get_pos_embedding(pos_data0, 128, device='cpu')
pos_data = pos_data0.unsqueeze(0)
pos_data = pos_data.expand(test_x_data.size(0), -1, -1)
test_x_data = torch.cat((test_x_data, pos_data), dim=2)


test_dataset = TensorDataset(test_x_data, test_y_data)
test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False)


model = model.to(device)

model.eval()


print("Analyzing test set...")

all_attention_maps = [] 
with torch.no_grad():
    prob_all_AUC = [] 
    label_all = []
    for _, (inputs, targets) in enumerate(tqdm(test_loader, desc="Test Processing")):
        
        get_local.clear()
        
        inputs = inputs.to(device)
        
        outputs, _ = model(inputs[:, :, :40], inputs[:, :, 40:])
        prob = torch.softmax(outputs, dim=1).cpu().detach().numpy()
        prob_all_AUC.extend(prob[:, 1])
        labels = targets.cpu()
        label_all.extend(labels)


        
        attention_maps = get_local.cache['Attention.forward'] # List of (batch_size, num_heads, seq_len, seq_len) for each layer
        
        attention_maps = [np.expand_dims(att, axis=1) for att in attention_maps] # Shape: (batch_size, 1, num_heads, seq_len, seq_len)
        
        attention_maps = np.concatenate(attention_maps, axis=1) # Shape: (batch_size, num_layers, num_heads, seq_len, seq_len)
        
        all_attention_maps.append(attention_maps)
    # Calculate AUC
    AUC = roc_auc_score(label_all, prob_all_AUC)
    print("test file AUC:{:.4f}".format(AUC))


sample_num = 0
final_avg_attn_per_layer_head = None


for attn_map in all_attention_maps:
    batch_size_num = attn_map.shape[0]
    attn_map = np.sum(attn_map, axis=0)
    attn_map = attn_map
    if final_avg_attn_per_layer_head is None:
        final_avg_attn_per_layer_head = attn_map
    else:
        final_avg_attn_per_layer_head += attn_map
    sample_num += batch_size_num


if sample_num > 0:
    num_layers = final_avg_attn_per_layer_head.shape[0]
    num_heads = final_avg_attn_per_layer_head.shape[1]
    seq_len = final_avg_attn_per_layer_head.shape[2] 
    final_avg_attn_per_layer_head /= sample_num

    num_features = seq_len



    print("\n--- Significant Features Extraction ---")

    significant_pairs, significant_nodes, aggregated_matrix, threshold = extract_significant_features(final_avg_attn_per_layer_head)

 
    sorted_pairs = sorted(significant_pairs.items(), key=lambda x: x[1], reverse=True)
    sorted_nodes = sorted(significant_nodes.items(), key=lambda x: x[1], reverse=True)

    for (i, j), weight in sorted_pairs[:20]:  
        print(f"({i},{j}): {weight:.4f}")


    for node, count in sorted_nodes[:50]:  
        print(f"SNP-{node}: {count}")



    data_list = []
    snp_name_file_path = './data/{}/fold_{}/snp_content_test_snp.txt'.format(data_name, args.fold)
    with open(snp_name_file_path) as f:
        for line in f:
            splits = line.split()
            if len(splits) > 2:
                column3 = splits[2]
                data_list.append(column3)
    assert len(data_list) == seq_len

    print("\n--- Significant Features Extraction ---")
    result_dir = f"result_{data_name}/result_{data_name}_fold_{args.fold}"
    os.makedirs(result_dir, exist_ok=True)
    save_path = os.path.join(result_dir, "snp_result.pkl")


    if os.path.exists(save_path):
        print(f"Loading precomputed results from {save_path}")
        with open(save_path, 'rb') as f:
            significant_pairs, significant_nodes = pickle.load(f)
    else:
  
        significant_pairs, significant_nodes, _, _ = extract_significant_features(final_avg_attn_per_layer_head)
        with open(save_path, 'wb') as f:
            pickle.dump((significant_pairs, significant_nodes), f)
        print(f"Saved results to {save_path}")


    print("\n--- Saving CSV Files ---")
    save_as_csv(significant_pairs, significant_nodes, data_list, result_dir)

    plt.figure(figsize=(12, 10))
    plt.imshow(np.log1p(aggregated_matrix), cmap='viridis', aspect='auto')
    plt.colorbar(label='Log-Scaled Attention')
    plt.title("Aggregated Attention Matrix")
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Index")
    plt.show()



    if sample_num > 0 and len(significant_pairs) > 0:
        plot_circular_network(significant_pairs, significant_nodes, data_list)
    else:
        print("No significant pairs to visualize.")

else:
    print("No batches processed for the test set.")