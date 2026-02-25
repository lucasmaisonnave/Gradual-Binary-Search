
import  utils
from hadamard_utils import random_hadamard_matrix
import torch
import matplotlib.pyplot as plt
from quant_utils import WeightQuantizer
import math

def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.
    
    Args:
    size (int): The size of the matrix (size x size).
    
    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q

def get_orthogonal_matrix(size, mode, device=utils.DEV):
    if mode == 'random':
        return random_orthogonal_matrix(size, device)
    elif mode == 'hadamard':
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f'Unknown mode {mode}')
    

def generate_concentrated_matrix(size, concentration_factor=10, sigma = 0.1, mu = 0):
    matrix = (torch.randn(size, size) * sigma + mu).to(utils.DEV)
    # Augmente certaines valeurs pour simuler une concentration
    matrix[0, 3] = concentration_factor  # Une colonne avec valeurs plus élevées
    # matrix[0, :] = concentration_factor  # Une ligne avec valeurs plus élevées
    return matrix

# Fonction pour appliquer une rotation orthogonale aléatoire
def apply_random_rotation(matrix, type = 'hadamard'):
    size = matrix.shape[0]
    Q = get_orthogonal_matrix(size, type).to(torch.float32)  # Génère une matrice orthogonale aléatoire

    return Q, matrix @ Q

def generate_values(type):
    SIZE= []
    ABS_MAX = []
    ABS_MAX_TH = []
    for i in range(6):
    # Paramètres
        size = 128 * 2**i  # Taille de la matrice
        SIZE.append(size)
        concentration_factor = 200  # Facteur de concentration

        # Générer la matrice concentrée
        original_matrix = generate_concentrated_matrix(size, concentration_factor)

        # Appliquer la rotation orthogonale aléatoire
        Q, rotated_matrix = apply_random_rotation(original_matrix, type)
        ABS_MAX.append(rotated_matrix.abs().max().item())
        if type == 'hadamard':
            ABS_MAX_TH.append(concentration_factor / math.sqrt(size))
        elif type == 'random':
            ABS_MAX_TH.append(concentration_factor * math.sqrt(2*math.log(size)) / math.sqrt(size))
    return SIZE, ABS_MAX, ABS_MAX_TH

# # Affichage des histogrammes pour comparer les distributions
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# # Distribution des valeurs avant rotation
# axes[0].hist(original_matrix.detach().cpu().numpy().flatten(), bins=256, color='skyblue', edgecolor='black')
# axes[0].set_title('Distribution avant rotation')
# axes[0].set_xlabel('Valeurs')
# axes[0].set_ylabel('Fréquence')

# # Distribution des valeurs après rotation
# axes[1].hist(rotated_matrix.detach().cpu().numpy().flatten(), bins=256, color='salmon', edgecolor='black')
# axes[1].set_title('Distribution après rotation')
# axes[1].set_xlabel('Valeurs')
# axes[1].set_ylabel('Fréquence')
SIZE_h, ABS_MAX_h, ABS_MAX_TH_h = generate_values('hadamard')
SIZE_r, ABS_MAX_r, ABS_MAX_TH_r = generate_values('random')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(SIZE_h, ABS_MAX_h, label = 'exp')
axes[0].plot(SIZE_h, ABS_MAX_TH_h, label = 'th')
axes[0].set_title(r'Hadamard $\frac{1}{\sqrt{n}}$')
axes[0].set_xlabel('Dimension')
axes[0].set_ylabel('Abs Max')
axes[0].grid()

axes[1].plot(SIZE_r, ABS_MAX_r, label = 'exp')
axes[1].plot(SIZE_r, ABS_MAX_TH_r, label = 'th')
axes[1].set_title(r'Random $\frac{\sqrt{2\log n}}{\sqrt{n}}$')
axes[1].set_xlabel('Dimension')
axes[1].set_ylabel('Abs Max')
axes[1].grid()

plt.tight_layout()
plt.legend()

plt.savefig('img/Max_Size.png')
plt.close()


# quant = WeightQuantizer()
# quant.configure(bits=4)

# error_quant = (original_matrix - (quant.quantize(rotated_matrix)) @ Q.T).abs().mean()
# print('Quantization error : {}'.format(error_quant))

# eigval = torch.linalg.eigvals(Q.T)
# fig, ax = plt.subplots(figsize=(12, 6))
# plt.scatter(torch.linspace(1, eigval.real.shape[0], eigval.real.shape[0]), eigval.real.abs().detach().cpu().numpy(), label='real')
# plt.scatter(torch.linspace(1, eigval.real.shape[0], eigval.real.shape[0]), eigval.imag.abs().detach().cpu().numpy(), label='imag')
# plt.legend(bbox_to_anchor=(1.3, 0.5), loc='center', fontsize=20)
# plt.tight_layout()
# plt.grid()
# plt.savefig('img/EigenValuesH.T.png')
# plt.close()