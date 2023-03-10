import os
import numpy as np
import torch
from torch import nn
import argparse
from tqdm import tqdm
from models import MolT5_smiles2text
from dataset import MolT5Dataset
from torch.utils.data import DataLoader
from metrics import calculate_bleu_score, calculate_rouge_score

def main(args):
    test_dataset = MolT5Dataset(os.path.join(args['data_path'], 'test.txt'))
    if args['debug']:
        test_dataset = test_dataset.__subset__(range(3*args['batch_size']))
    test_dataloader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False)
    model = MolT5_smiles2text(option=args['option'], max_len=args['max_len'])
    model.load_state_dict(torch.load(os.path.join(args['save_path'], 'model_best.pt')))

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('num_params: {}'.format(num_params))

    model.eval()
    c_list = []
    pred_list, label_list = [], []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            cid_list, smiles_list, description_list = batch
            pred = model.generate(smiles_list)
            c_list.extend(cid_list)
            pred_list.extend(pred)
            label_list.extend(description_list)
    
    # calculate the bleu score
    bleu_metrics = ['bleu-2', 'bleu-4']
    bleu_scores = calculate_bleu_score(pred_list, label_list, metrics=bleu_metrics)
    for metric in bleu_metrics:
        print('{} score: {}'.format(metric, bleu_scores[metric]))

    # calculate the rouge score
    rouge_metrics = ['rouge-l', 'rouge-1', 'rouge-2']
    rouge_scores = calculate_rouge_score(pred_list, label_list, metrics=rouge_metrics)
    for metric in rouge_metrics:
        print('{} score: {}'.format(metric, rouge_scores[metric]))
    
    # sort the results
    bleu_score = bleu_scores[bleu_metrics[0]]
    bleu_score, pred_list, label_list = zip(*sorted(zip(bleu_score, pred_list, label_list), reverse=True))

    # save the results to file
    with open(os.path.join(args['save_path'], 'test_results.txt'), 'w') as f:
        for cid, pred, label in zip(c_list, pred_list, label_list):
            f.write(cid + '\t' + pred + '\t' + label + '\n')
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--option', type=str, default='t5-base')
    parser.add_argument('--data_path', type=str, default='/data/xuhw/Courses/NLP/MolT5_reimplementation/data/')
    parser.add_argument('--save_path', type=str, default='/data/xuhw/Courses/NLP/MolT5_reimplementation/results/test/model/')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--warmup_steps', type=int, default=4000)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    args = args.__dict__
    main(args)