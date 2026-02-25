import torch
import json
from eval_utils.main import ptq_model, update_quantizer
from utils import data_utils, eval_utils, utils
import os
from copy import deepcopy

cache_dir = '/data1/is156025/lm270675/.cache/huggingface/hub'
n_layers = {'meta-llama/Llama-2-7b-hf': 32,
'meta-llama/Llama-3.2-1B' : 16,
'meta-llama/Llama-3.1-8B' : 32,
'meta-llama/Llama-2-13b-hf': 40, 
'mistralai/Mistral-7B-v0.1' : 32,
'mistralai/Mistral-7B-Instruct-v0.3': 32,
'meta-llama/Llama-2-7b-hf': 32, 
'meta-llama/Meta-Llama-3-8B' : 32,
'facebook/opt-13b': 40,
'bigscience/bloom-7b1' : 32,
'deepseek-ai/DeepSeek-R1-Distill-Llama-8B':32,
'Qwen/Qwen2.5-7B-Instruct' : 28,
'Qwen/Qwen2.5-1.5B-Instruct' : 28,
'Qwen/Qwen2.5-0.5B-Instruct' : 24,
'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' : 28,
'microsoft/Phi-4-mini-instruct': 28
}



class GBS():
    def __init__(self, ptq_args, model_args, model, tokenizer, b = [8, 6, 4], alpha = 0.1):

        # On va quantifier le modèle à chaque nouvelle configuration
        # La fonction quantize_model doit donc prendre en compte bit, max ainsi que l'indice de la couche à modifier
        self.model_name = model_args.input_model
        self.b = b
        self.alpha = alpha
        self.n_layers = n_layers[model_args.input_model]
        self.folder = './grid_search/' + self.model_name + '/'
        self.filename = 'rotate' if ptq_args.rotate else '' 
        self.filename +=  '_W_RTN' if ptq_args.w_rtn else '_W_GPTQ' 
        self.filename += '_W' + str(ptq_args.w_bits) + 'A' + str(ptq_args.a_bits) + 'KV' + str(ptq_args.v_bits) 
        self.filename_noexp = self.filename + ".json"
        self.filename += ".json"
        self.args = ptq_args
        self.world_size = torch.cuda.device_count()

        print('Loading Test')
        self.testloader = data_utils.get_wikitext2(
            seed=ptq_args.seed,
            seqlen=2048,
            eval_mode=True,
            tokenizer=tokenizer
        )
        print('Loading Train')
        self.trainloader = data_utils.get_wikitext2(
            seed=ptq_args.seed,
            seqlen=2048,
            tokenizer=tokenizer,
            eval_mode=False,
            gbs=True
        )
        size = int(alpha * self.trainloader.input_ids.shape[-1])
        self.trainloader.input_ids = self.trainloader.input_ids[:,:size].cuda()

        self.model = model


        self.epsilon = 1e-3
        self.max_iterations = 10

    
    def append_to_json(self, file_name, data):
        try:
            # Read existing data from the file
            if not os.path.exists(self.folder):
                os.makedirs(self.folder)
            with open(file_name, 'r') as file:
                existing_data = json.load(file)
        except FileNotFoundError:
            # If the file doesn't exist, start with an empty list
            existing_data = []
            
        except json.JSONDecodeError:
            # If the file is empty or not valid JSON, start with an empty list
            existing_data = []
        
                # Append the new data
        existing_data.append(data)

        # Write the updated data back to the file
        with open(file_name, 'w') as file:
            json.dump(existing_data, file, indent=4)
    
    def f(self, config, eval = False):
        update_quantizer(self.args, self.model, config)
        # if self.prefixed_key_values is not None:
        #     self.prefixed_key_values = p_model_utils.mv_kv_cache(self.prefixed_key_values, self.model)
        self.model.eval()
        torch.cuda.empty_cache()
        test_ppl = None
        train_ppl = None
        if eval:
            test_ppl = eval_utils.evaluator(self.model, self.testloader, utils.DEV, self.args) #, prefixed_key_values)
        else:
            torch.cuda.empty_cache()
            train_ppl = eval_utils.evaluator(self.model, self.trainloader, utils.DEV, self.args) #, prefixed_key_values)

        return train_ppl, test_ppl

    def resume(self):
        with open(self.folder + self.filename, 'r') as file:
            data = json.load(file)
            return data[-1]['config']
        
    def binary_search(self, list_max_bit, i, proj, l, bit, best_max_bit):
        a = 0
        b = 1
        iterations = 0
        m = (a + b) / 2
        

        list_max_bit[i][proj[l]]['max'] = m
        list_max_bit[i][proj[l]]['bit'] = bit

        config = {'max_bit': list_max_bit, 'layer': i,  'bit': bit, 'proj': proj[:l + 1]}
        print("m = {:.2f}".format(m))
        fm, ftest = self.f(config)
        results = {'config': config, 'ppl': {'train': fm, 'test': ftest}}
        self.append_to_json(self.folder + self.filename, results)
        
        # Recherche dichotomique
        while b - a > self.epsilon and iterations < self.max_iterations:
            if iterations % 2 == 0:
                x = (a + m) / 2
            else:
                x = (m + b) / 2
    
            list_max_bit[i][proj[l]]['max'] = x
            list_max_bit[i][proj[l]]['bit'] = bit

            config = {'max_bit': list_max_bit, 'layer': i, 'bit': bit, 'proj': proj[:l + 1]}
            print("m = {:.2f}".format(x))
            fx, fxtest = self.f(config)

            results = {'config': config, 'ppl': {'train': fx, 'test': fxtest}}

            self.append_to_json(self.folder + self.filename, results)
            
            if fx < fm:
                if x < m:
                    b = m
                else:
                    a = m
                m, fm, ftest = x, fx, fxtest
            else:
                if x < m:
                    a = x
                else:
                    b = x
            
            iterations += 1
        best_max_bit[i][proj[l]]['max'] = m
        best_max_bit[i][proj[l]]['bit'] = bit
        f_best_train = fm
        f_best_test = ftest
        list_max_bit = deepcopy(best_max_bit)
        return list_max_bit, best_max_bit, f_best_test, f_best_train, config


    def search(self):

        print('Starting Search')
        if self.args.resume_gs:
            with open(self.folder + self.filename, 'r') as file:
                existing_data = json.load(file)
                # remove last proj optimisation from file
                resume_p = len(existing_data[-1]['config']['proj']) - 1
                resume_l = existing_data[-1]['config']['layer']
                while len(existing_data[-1]['config']['proj']) - 1 == resume_p:
                    existing_data.pop()

                last_data = existing_data[-1]

                resume_config = last_data['config']['max_bit']
                resume_ppl = last_data['ppl']['train']
                with open(self.folder + self.filename, 'w') as file:
                    json.dump(existing_data, file, indent=4)
                print("Resume Grid Search from : layer {} proj {}".format(resume_l, resume_p))
        else:
            resume_l = 0
            resume_p = 0
            
        for bit in self.b:
            proj = ['q_proj', 'k_proj', 'v_proj', 'qk_rotation', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'] # 
            tmp = {}
            base_tmp = {'max': 1, 'bit': 16}
            for p in proj:
                tmp[p] = deepcopy(base_tmp)
            if self.args.resume_gs:
                list_max_bit = resume_config
                best_max_bit = deepcopy(resume_config)
                f_best_train = resume_ppl
            else:
                list_max_bit = [deepcopy(tmp) for _ in range(self.n_layers)]
                best_max_bit = [deepcopy(tmp) for _ in range(self.n_layers)]
                f_best_train = 1e5
            f_best_test = 1e5
            for i in range(resume_l,self.n_layers):
                print("Layer " + str(i))
                for l in range(resume_p, len(proj)):
                    print("Proj " + proj[l])
                    list_max_bit, best_max_bit, f_best_test, f_best_train, config = self.binary_search(list_max_bit, i, proj, l, bit, best_max_bit)
                resume_p = 0
            config['max_bit'] = deepcopy(best_max_bit)
            if f_best_test is None:
                _, f_best_test = self.f(config, eval = True)
            best_config_bit = {'config': config, 'ppl': {'train': f_best_train, 'test': f_best_test}}  
            self.append_to_json(self.folder + 'best_config_' + self.filename, best_config_bit)
        
        return best_max_bit


