#!/bin/bash

declare -A user_gpu_map

# Récupère les PIDs, GPU index, et utilisateurs
while IFS=, read -r gpu_index pid _; do
    pid=$(echo $pid | xargs)  # remove leading/trailing spaces
    gpu_index=$(echo $gpu_index | xargs)
    user=$(ps -o user= -p "$pid" 2>/dev/null)

    if [[ -n "$user" ]]; then
        key="$user|$gpu_index"
        user_gpu_map["$key"]=1
    fi
done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader,nounits)

# Compter combien de GPU différents chaque utilisateur utilise
declare -A user_gpu_count

for key in "${!user_gpu_map[@]}"; do
    user="${key%%|*}"
    ((user_gpu_count["$user"]++))
done

# Afficher le résultat
echo "Utilisateur | Nombre de GPU utilisés"
echo "----------- | ----------------------"
for user in "${!user_gpu_count[@]}"; do
    echo "$user | ${user_gpu_count[$user]}"
done
