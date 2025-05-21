import json
import gradio as gr
import tempfile
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os
import numpy as np
import torchaudio
import functools
import argparse
import yaml
from generation import decode_wave, GenerationAudioTokens, hifi_gan as hifi_gan_model
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import math
from eval_story import process_directory_story,plot_and_save_results_from_file
from eval_cmmlu import process_directory as process_directory_cmmlu
from eval_cmmlu4 import process_directory as process_directory_cmmlu4
import glob
from tqdm import tqdm
import sys

class jsonlDataset(Dataset):
    def __init__(self, path):
        self.data = []
        with open(path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return d

def proc_dataset(dataset_name,dataloader,data_path,answer_list,key=None):

    for batch in tqdm(dataloader, desc="Processing batches", unit="batch"):
        if isinstance(batch['uttid'], torch.Tensor):
            uttids = [d.item() for d in batch['uttid']]
        else:
            uttids = [d for d in batch['uttid']]

        questions = [d for d in batch['first']]
        audio_paths = [os.path.join(data_path,str(d) + '.wav') for d in uttids]

        if 'tory' in dataset_name:
            answers = [d for d in batch['last']]
        else:
            answers = [d for d in batch[key]]
        losses,ppls = generate_batch_response(lang, model, questions, answers) 
        for idx in range(len(uttids)):
            answer_data = {
                "uttid": uttids[idx],
                "key": key,
                "loss": [losses[idx]],
                "ppl": [ppls[idx]],
            }

            answer_list.append(answer_data)
    return answer_list


def create_position_ids_from_input_ids_left_padded(input_ids, attention_mask, past_key_values_length=0):

    seq_lengths = attention_mask.sum(dim=1)
    position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
    for i in range(input_ids.size(0)):
        actual_seq_length = seq_lengths[i].item()
        if actual_seq_length > 0:
            position_ids[i, -actual_seq_length:] = torch.arange(
                past_key_values_length, past_key_values_length + actual_seq_length, 
                dtype=torch.long, device=input_ids.device
            )

    return position_ids

def init_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model.eval()
    model.bind_processor(tokenizer, training=False, relative_path='/', default_client='oss')
    cuda_hifi_gan = hifi_gan_model.eval().cuda()
    cuda_hifi_gan.device = 'cuda'
    return model, tokenizer, cuda_hifi_gan

def mask_loss_ppl(model,ret,labels,input_ids):

    _, sp_mask, _ = model.model.get_multimodal_mask(input_ids, model.config.audio_config.audio_pad_token_id, model.config.multimodal_special_token_list)
    special_with_loss_list = list(set(model.config.multimodal_special_token_list) - set(model.config.multimodal_special_token_no_loss_list))
    _, sp_with_loss_mask, lm_head_mask = model.model.get_multimodal_mask(input_ids, model.config.audio_config.audio_pad_token_id, special_with_loss_list)


    shift_labels = torch.nn.functional.pad(labels[:,1:], (0, 1), value=0)
    shift_input_ids = torch.nn.functional.pad(input_ids[:,1:], (0, 1), value=0)
    audio_trainable_mask = ((shift_labels == model.config.audio_config.audio_pad_token_id) | (shift_labels == model.config.audio_config.audiogen_end_token_id)).to(ret.logits.device)
    
    # Shift so that tokens < n predict n
    shift_logits = ret.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    valid_mask = torch.gt(shift_labels, -1)  
    sp_mask = sp_mask[..., 1:].contiguous()
    sp_with_loss_mask = sp_with_loss_mask[..., 1:].contiguous()
    text_mask = torch.logical_and(valid_mask, torch.logical_not(sp_mask))
    valid_mask = torch.logical_or(torch.logical_and(valid_mask, torch.logical_not(sp_mask)),sp_with_loss_mask)

    # Flatten the tokens
    shift_logits = shift_logits.view(-1, model.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    flatten_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction='none')
    
    losses = flatten_loss.view(labels.shape[0],-1)
    losses_sum = [loss_list.sum() for loss_list in losses]
    len_sum = [mask.sum() + + 1e-10 for mask in valid_mask]

    lossed_mean = [(loss / length).item() for loss, length in zip(losses_sum, len_sum)]
    losses_list = lossed_mean
    ppls = [math.exp(loss) for loss in losses_list]
    return losses_list,ppls

def generate_response(lang,model,input_audio_path, answer):
    if lang == "zh":
        input_string = '下面是一段语音和文本交错的内容：'
    else:
        input_string = 'Below is a segment of alternating speech and text content: '

    input_string = input_string + '<audio_start_baichuan>{\"path\": \"%s\"}<audio_end_baichuan>'%input_audio_path 
    input_string = input_string + '<trainable_start>'+ answer +'<trainable_end>'

    pret = model.processor([input_string])
    textlen = pret.input_ids.shape[1]
    batch_data = pret
    ret = model(input_ids=batch_data.input_ids.cuda(), labels=batch_data.labels.cuda(), 
                audios=batch_data.audios.cuda(), encoder_length=batch_data.encoder_length.cuda(), bridge_length=batch_data.bridge_length.cuda(),
                        )

    return float(ret.loss)

def generate_batch_response(lang, model, questions, answers):
    
    batch_input_strings = []

    for question, answer in zip(questions, answers):

        if lang == "zh":
            input_string = question + '<trainable_start>' + answer + '<trainable_end>'
        else:
            input_string = question + '<trainable_start> ' + answer + '<trainable_end>'
        batch_input_strings.append(input_string)

    batch_data = model.processor(batch_input_strings)
    batch_position_ids = create_position_ids_from_input_ids_left_padded(batch_data.input_ids,batch_data.attention_mask)

    labels = batch_data['labels']
    input_ids=batch_data['input_ids']

    ret = model(
        input_ids=batch_data['input_ids'].cuda(),
        attention_mask=batch_data.attention_mask.cuda(),
        position_ids = batch_position_ids.cuda(),
        labels=batch_data.labels.cuda(), 
    )

    losses_list,ppls = mask_loss_ppl(model,ret,labels,input_ids)

    return losses_list,ppls

 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_list", nargs='+', default=[], help="List of datasets")
    parser.add_argument("--checkpoint",default='/checkpoint_load/ckpt_*_*_*/')
    parser.add_argument("--root_dir",default='/global_data/')
    parser.add_argument("--config",default='eval_s2t.yaml')
    parser.add_argument("--result_dir",default='results2/')
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--plot", action='store_true', help="plots result (default: False)")
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()

    with open(args.config, encoding="utf8") as f:
        config = yaml.full_load(f)
    root_dir = args.root_dir
    dataset_list = args.dataset_list 
    model_paths = glob.glob(args.checkpoint)
    dataset_paths = []
    langs = []
    audio_paths = []
    dataloader_list = []


    for dataset_name in dataset_list:
        if dataset_name in config["datasets"]:
            path = os.path.join(root_dir, config["datasets"][dataset_name]["path"])
            audio_path = os.path.join(root_dir, config["datasets"][dataset_name]["audio_path"])
            lang = config["datasets"][dataset_name]["lang"]
        else:
            raise "Not support this dataset for eval !!!"
        dataset_paths.append(path)
        langs.append(lang)
        audio_paths.append(audio_path)
        dataloader_list.append(DataLoader(jsonlDataset(dataset_paths[-1]), batch_size=args.batch_size, shuffle=False))

    #模型文件
    for ckpt in model_paths:
        model, tokenizer, cuda_hifi_gan = init_model(ckpt)
        for i in range(len(dataset_list)):
            dataset_name = dataset_list[i]
            audio_path = audio_paths[i]
            dataloader = dataloader_list[i]
            answer_dir = os.path.join(args.result_dir, dataset_name, os.path.basename(ckpt.rstrip('/')))
            os.makedirs(answer_dir, exist_ok=True)
            answer_list = []

            if dataset_name == 'cmmlu_write_4':
                keys = ['A','B','C','D']
                for key in keys:
                    answer_list = proc_dataset(dataset_name,dataloader,audio_path,answer_list,key=key)
            else:
                answer_list = proc_dataset(dataset_name,dataloader,audio_path,answer_list,key=None)


            #result
            output_file = os.path.join(answer_dir, 'result.jsonl')
            with open(output_file, "w", encoding="utf-8") as jsonl_file:
                for entry in answer_list:
                    jsonl_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

            print(f"JSONL文件已保存到: {output_file}")
            out_dir = os.path.join(args.result_dir, dataset_name)
            result_txt = os.path.join(out_dir, 'results_summary.txt')

            if dataset_name == 'cmmlu_write_4':
                process_directory_cmmlu4(out_dir,result_txt,dataset_paths[i])
            else:
                process_directory_story(out_dir,result_txt)
            print(f"Results have been saved to {result_txt}")
            with open(result_txt, 'r', encoding='utf-8') as file:
                print('Dataset Name: ',dataset_name)
                for line in file:
                    print(line.strip())  
            if args.plot:
                plot_and_save_results_from_file(result_txt, out_dir)







