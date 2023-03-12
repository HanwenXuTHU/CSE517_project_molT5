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
    model.load_state_dict(torch.load(os.path.join(args['save_path'], 'model_best.pt'), map_location=torch.device('cuda:0')))

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('num_params: {}'.format(num_params))

    model.eval()
    c_list = []
    pred_list, label_list = [], []
    pp_list, ll_list = [], []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            cid_list, smiles_list, description_list = batch
            # print('len(cid_list): {}'.format(len(cid_list)))
            # print('len(smiles_list): {}'.format(len(smiles_list)))
            # print('len(description_list): {}'.format(len(description_list)))
            pred = model.generate(smiles_list)
            c_list.extend(cid_list)
            pred_list.extend(pred)
            label_list.extend(description_list)

            for pp, ll in zip(pred, description_list):
                pp_list.append(model.tokenizer(pp)['input_ids'])
                ll_list.append(model.tokenizer(ll)['input_ids'])
                # print(pp_list[-1])
                # print(ll_list[-1])
    
    # save the results to file
    with open(os.path.join(args['save_path'], '{}_results.txt'.format('debug' if args['debug'] else 'test')), 'w') as f:
        for cid, pred, label, pp, ll in zip(c_list, pred_list, label_list, pp_list, ll_list):
            f.write(cid + '\t' + pred + '\t' + label + '\n')
            # f.write(cid + '\t' + pred + '\t' + label + '\t' + pp + '\t' + ll + '\n')
    f.close()
    
    # calculate the bleu score
    bleu_metrics = ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4']
    # bleu_scores = calculate_bleu_score(pred_list, label_list, metrics=bleu_metrics)
    bleu_scores = calculate_bleu_score(pp_list, ll_list, metrics=bleu_metrics)
    for metric in bleu_metrics:
        print('{} score: {}'.format(metric, np.mean(bleu_scores[metric])))

    # calculate the rouge score
    rouge_metrics = ['rouge-l', 'rouge-1', 'rouge-2']
    # rouge_scores = calculate_rouge_score(pred_list, label_list, metrics=rouge_metrics)
    rouge_scores = calculate_rouge_score(pred_list, label_list, metrics=rouge_metrics)
    for metric in rouge_metrics:
        print('{} score: {}'.format(metric, np.mean(rouge_scores[metric])))
    
    print('len(c_list): {}'.format(len(c_list)))
    print('len(pred_list): {}'.format(len(pred_list)))
    print('len(label_list): {}'.format(len(label_list)))
    print('len(pp_list): {}'.format(len(pp_list)))
    print('len(ll_list): {}'.format(len(ll_list)))
    
    # sort the results
    bleu_score = bleu_scores[bleu_metrics[-1]]
    # bleu_score, c_list, pred_list, label_list = zip(*sorted(zip(bleu_score, c_list, pred_list, label_list), reverse=True))
    bleu_score, c_list, pred_list, label_list, pp_list, ll_list = zip(*sorted(zip(bleu_score, c_list, pred_list, label_list, pp_list, ll_list), reverse=True))

    print('len(bleu_score): {}'.format(len(bleu_score)))

    # save the results to file
    file_name = '{}_results.txt'.format('debug' if args['debug'] else 'smiles2text_{}'.format(args['option'].split('-')[-1]))
    with open(os.path.join(args['save_path'], file_name), 'w') as f:
        for bscore, cid, pred, label, pp, ll in zip(bleu_score, c_list, pred_list, label_list, pp_list, ll_list):
            f.write('{:.04f}'.format(bscore) + '\t' + cid + '\t' + pred + '\t' + label + '\n')
            # f.write(cid + '\t' + pred + '\t' + label + '\t' + pp + '\t' + ll + '\n')
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