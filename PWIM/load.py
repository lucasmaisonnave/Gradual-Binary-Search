from ShortGPT.short_gpt.short_hf import ShortHFModel
import os
def load_short_model(model_name, N, dataset):
    short_model = ShortHFModel(
        model_name=model_name,
        layers_path="model.layers",
        n_prune_layers=9
    )

    n_layers = len(short_model.layers)
    ratio = N / float(n_layers)
    layers_to_remove = [n_layers - 2 - i for i in range(int(ratio * n_layers))]
    short_model.remove_layers(layers_to_remove)

    local_dir = "/data1/is156025/lm270675/meta-labo/EAH-ViT/LLM/ShortGPT/short_gpt/models/" + model_name + "/" + dataset + "/N_{}".format(N)
    if not os.path.exists(local_dir):
        assert "Model doesn't not exist : " + model_name + "/" + dataset + "/N_{}".format(N)
    short_model.model.load_adapter(local_dir)

    # reassign layer_idx to attentions for caching
    for layer_idx, module in enumerate(short_model.layers):
        module.self_attn.layer_idx = layer_idx

    tokenizer = short_model.tokenizer
    model = short_model.model

    return tokenizer, model