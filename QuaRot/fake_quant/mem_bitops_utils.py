import pandas as pd
from main import load_rotate_quantize_model
import torch.nn as nn
import utils
from quant_utils import cleanup_model
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
import math
import model_utils

def get_bitops_memory(args):
    model = model_utils.get_model(args.model, args.hf_token, args.cache_dir)
    model,_ = load_rotate_quantize_model(args, model)
    seqlen = model.seqlen
    bitops = 0
    bit = args.bit
    n = len(model.model.layers)
    mem = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            dim_in = module.in_features
            dim_out = module.out_features
            if 'lm_head' in name:
                bitops += seqlen * dim_in**2 * dim_out * 16**2
                mem += dim_in * dim_out * 16
            else:
                bitops += seqlen * dim_in**2 * dim_out * bit**2
                mem += dim_in * dim_out * bit
    
    # Produit matriciel de la carte attentionnel, k est quantifé en 4 bits
    dim = 4096 + args.expand
    bitops += n * seqlen**2 * dim**2 * bit * 16
    cleanup_model(model)
    del model
    torch.cuda.empty_cache()
    return bitops, mem

def update_csv(csv_file, output_file):
    # Lecture du fichier CSV
    df = pd.read_csv(csv_file)
    
    # Vérifier que les colonnes nécessaires existent
    colonnes_attendues = {'model', 'bit', 'expand'}
    if not colonnes_attendues.issubset(df.columns):
        raise ValueError(f"Le fichier CSV doit contenir les colonnes : {colonnes_attendues}")
    
    # Initialiser les colonnes 'mem' et 'bitop' si elles n'existent pas
    if 'mem' not in df.columns:
        df['mem'] = None
    if 'bitops' not in df.columns:
        df['bitops'] = None
    
    # Parcours de chaque ligne et mise à jour
    for idx, row in df.iterrows():
        if pd.isna(df.at[idx, 'bitops']):
            args.model = row['model']
            args.expand = row['expand']
            args.bit = row['bit']
            
            # Appel de la fonction pour obtenir mem et bitops
            bitops, mem = get_bitops_memory(args)
            
            # Mise à jour des colonnes
            df.at[idx, 'mem'] = mem
            df.at[idx, 'bitops'] = bitops
        
            # Sauvegarde du fichier CSV mis à jour
            df.to_csv(output_file, index=False)
            print(f"Fichier CSV mis à jour et sauvegardé sous '{output_file}'.")

def plot_data(csv_file):
    # Lecture du fichier CSV
    df = pd.read_csv(csv_file)
    
    # Vérification que les colonnes nécessaires sont présentes
    expected_columns = {'bitops', 'ppl', 'mem', 'model', 'grid_search'}
    if not expected_columns.issubset(df.columns):
        raise ValueError(f"Le fichier CSV doit contenir les colonnes : {expected_columns}")
    
    # Facteur d'échelle pour la taille des marqueurs
    scale_factor = 0.4e-8  # à ajuster selon l'échelle de vos données
    
    # Récupération des modèles uniques et attribution d'une couleur à chacun
    unique_models = df['model'].unique()
    names = unique_models.copy()
    for i in range(unique_models.shape[0]):
        names[i] = unique_models[i].split('/')[-1]

    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_models)))
    
    plt.figure(figsize=(12, 6))
    
    # Parcours des modèles et affichage des points selon grid_search
    for i, model in enumerate(unique_models):
        df_model = df[df['model'] == model]
        
        # Points pour lesquels grid_search est False : contour blanc
        df_no_gs = df_model[df_model['grid_search'] == False]
        if not df_no_gs.empty:
            plt.scatter(
                df_no_gs['bitops'],
                df_no_gs['ppl'],
                s=df_no_gs['mem'] * scale_factor,
                alpha=0.6,
                edgecolors='w',
                color=colors[i]
            )
        
        # Points pour lesquels grid_search est True : contour rouge, tout en restant plein
        df_gs = df_model[df_model['grid_search'] == True]
        if not df_gs.empty:
            plt.scatter(
                df_gs['bitops'],
                df_gs['ppl'],
                s=df_gs['mem'] * scale_factor,
                alpha=0.6,
                edgecolors='red',
                color=colors[i]
            )
    
    plt.xlabel('Bitops')
    plt.ylabel('PPL')
    plt.title('PPL en fonction des Bitops\n(taille des points = memory)')
    plt.grid(True, which="both", ls="--")
    
    # Application d'une échelle logarithmique sur les deux axes
    plt.xscale('log')
    plt.yscale('log')
    
    # Légende pour les modèles, placée à l'extérieur du graphique
    model_handles = []
    for i, model in enumerate(names):
        handle = mlines.Line2D([], [], marker='o', color=colors[i], linestyle='None',
                               markersize=10, label=model)
        model_handles.append(handle)
    legend_models = plt.legend(handles=model_handles, title='Modèle', loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Légende pour l'indicateur grid_search
    no_gs_handle = mlines.Line2D([], [], marker='o', color='black', linestyle='None', markersize=10,
                                 markerfacecolor='grey', markeredgecolor='w', label='False')
    gs_handle = mlines.Line2D([], [], marker='o', color='black', linestyle='None', markersize=10,
                              markerfacecolor='grey', markeredgecolor='red', label='True')
    legend_gs = plt.legend(handles=[no_gs_handle, gs_handle], title='Grid Search', loc='lower left', bbox_to_anchor=(1.05, 0.3))
    
    # Ajout de la légende des modèles à la figure afin de conserver les deux
    plt.gca().add_artist(legend_models)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    
    plt.savefig('test.png')


def dimension_equivalent(n, N, A, D):
    return math.sqrt(1./N * ((n + D)**2 * A).sum()) - n 

if __name__ == "__main__":
    args = utils.parser_gen()
    D = np.array([0, 124, 268, 412, 544, 688, 824, 956, 1084, 1208, 1336, 1468, 1588, 1712, 1844, 1972, 2104, 2228, 2356, 2476, 2608, 2732, 2864, 2984, 3112, 3236, 3364, 3496, 3628, 3772, 3916, 4052])
    A = 1
    print(dimension_equivalent(4096, 32, A, D))
    args.rotate = True
    args.k_group_size = 128
    args.w_clip = True
    args.a_bits = 16
    args.v_bits = 16
    args.k_bits = 16
    args.w_bits = 16
    # Nom du fichier d'entrée et de sortie
    input_csv = "./fake_quant/data_mem.csv"        # Remplacez par le chemin de votre fichier CSV
    #update_csv(input_csv, input_csv)
    plot_data(input_csv)
