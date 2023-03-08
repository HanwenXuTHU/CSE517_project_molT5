import os
import torch
import argparse
from torch.utils.data import DataLoader

from tqdm import tqdm
from models import MolT5_text2smiles
from dataset import MolT5Dataset

import logging
from torch.utils.tensorboard import SummaryWriter
from transformers.optimization import get_linear_schedule_with_warmup


def main(args):
    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])
    # create logger and save it to the save_path
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(os.path.join(args['save_path'], 'log.txt'))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    device = torch.device('cuda:{}'.format(args['device']))

    train_dataset = MolT5Dataset(os.path.join(args['data_path'], 'train.txt'))
    val_dataset = MolT5Dataset(os.path.join(args['data_path'], 'validation.txt'))
    print('len(train): {}, len(val): {}'.format(len(train_dataset), len(val_dataset)))

    train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)

    model = MolT5_text2smiles(option=args['option'], max_len=args['max_len'])
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    # use warmups linear scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=args['warmup_steps'], 
                                                num_training_steps=len(train_dataloader)*args['epoch'])

    best_loss = 1e10

    # write the training loss and validation loss to the tensorboard
    writer = SummaryWriter(os.path.join(args['save_path'], 'tensorboard'))

    for epoch in range(args['epoch']):
        model.train()
        pbar = tqdm(train_dataloader)
        train_loss = 0
        for batch in pbar:
            optimizer.zero_grad()
            cid_list, smiles_list, description_list = batch
            loss, _ = model(description_list, smiles_list)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()/len(train_dataloader)
            pbar.set_description('Epoch: {}, Loss: {}'.format(epoch, loss.item()))
            scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc='Validation'):
                cid_list, smiles_list, description_list = batch
                loss, _ = model(description_list, smiles_list)
                val_loss += loss.item()/len(val_dataloader)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args['save_path'], 'model_best.pt'))
            logger.info('save model to {}'.format(os.path.join(args['save_path'], 'model_best.pt')))
        logger.info('Epoch: {}, Loss: {}, Best Loss: {}'.format(epoch, loss.item(), best_loss))

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
    writer.close()


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
    args = parser.parse_args()
    args = args.__dict__
    main(args)
