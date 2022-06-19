import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch, random
from tqdm import trange
import numpy as np

from dataset import TaggingDataset
from utils import Vocab
from model import TaggingClassifier, SlotRNN
# from torch.utils.tensorboard import SummaryWriter

# https://www.analyticsvidhya.com/blog/2020/01/first-text-classification-in-pytorch/
# https://towardsdatascience.com/multiclass-text-classification-using-lstm-in-pytorch-eac56baed8df

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

# model_name = 'slot_-3_02'
# logger = SummaryWriter(log_dir=f'log/slot/{model_name}')
seed = 567
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

def main(args):
    with open(args.cache_dir / "slot_vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    # datasets: Dict[str, TaggingDataset] = {
    #     split: TaggingDataset(split_data, vocab, tag2idx, args.max_len)
    #     for split, split_data in data.items()
    # }
    datasets = {}
    datasets[SPLITS[0]] = TaggingDataset(data[SPLITS[0]], vocab, tag2idx)
    datasets[SPLITS[1]] = TaggingDataset(data[SPLITS[1]], vocab, tag2idx, datasets[SPLITS[0]].max_len)

    train_data_loader = torch.utils.data.DataLoader(datasets['train'],
                                        batch_size = args.batch_size,
                                        shuffle=True,
                                        num_workers=0,
                                        collate_fn=datasets['train'].collate_fn,
                                        )

    eval_data_loader = torch.utils.data.DataLoader(datasets['eval'],
                                        batch_size = args.batch_size,
                                        shuffle=True,
                                        num_workers=0,
                                        collate_fn=datasets['eval'].collate_fn,
                                        )

    embeddings = torch.load(args.cache_dir / "slot_embeddings.pt")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_class = datasets['train'].num_classes + 1
    # model = TaggingClassifier(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, num_class, datasets[SPLITS[0]].max_len)
    
    model = SlotRNN(embeddings, args.hidden_size, num_class, args.bidirectional)

    # model = Slot(embeddings, args.hidden_size, num_class, device)
    
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=9).to(device)
    criterion.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    previous_accuracy = 0
    # epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in range(args.num_epoch):
        model.train()
        correct = 0
        train_loss = []
        test_loss = []

        for i, (targets, label, length) in enumerate(train_data_loader):
            targets = targets.to(device)
            label = label.to(device).long()

            outputs = model(targets).float()
            losses = criterion(outputs.reshape(-1, num_class), label.reshape(-1))

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            pred = torch.argmax(outputs, dim=2).reshape(-1, datasets[SPLITS[0]].max_len)

            label = label.cpu().numpy()
            pred = pred.cpu().numpy()

            correct += np.sum([ (p[:le]==la[:le]).all() for (p, la, le) in zip(pred, label, length) ])

            # print(f'Batch: {i}/{int(len(train_data_loader.dataset)/args.batch_size)}, Loss: {np.mean(train_loss):.3f}, Training Correct: {correct}')

            train_loss.append(losses.item())
            # logger.add_scalar('training loss',
            #                     losses,
            #                     epoch * len(train_data_loader) + i + 1)

        train_accuracy = correct / len(train_data_loader.dataset)
        correct = 0
        with torch.no_grad():
            for i, (targets, label, length) in enumerate(eval_data_loader):
                targets = targets.to(device)
                label = label.to(device).long()

                outputs = model(targets).float()

                pred = torch.argmax(outputs, dim=2).reshape(-1, datasets[SPLITS[0]].max_len)

                label = label.cpu().numpy()
                pred = pred.cpu().numpy()

                correct += np.sum([ (p[:le]==la[:le]).all() for (p, la, le) in zip(pred, label, length) ])

        eval_accuracy = correct / len(eval_data_loader.dataset)
        print(f'Epoch: {epoch}/{args.num_epoch}, Loss: {np.mean(train_loss):.3f}, Training Accuracy: {train_accuracy:.3f}, Eval Accuracy: {eval_accuracy:.3f}')
        
        # logger.add_scalar('train accuracy',
        #                         train_accuracy,
        #                         epoch * len(train_data_loader) + i + 1)

        # logger.add_scalar('eval accuracy',
        #                         eval_accuracy,
        #                         epoch * len(train_data_loader) + i + 1)

        if previous_accuracy <= eval_accuracy:
            print(f'save {args.ckpt_dir}/latest_2.pth')
            torch.save({'epoch': epoch, 
                    'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': losses,}, f'{args.ckpt_dir}/slot_latest.pth')
            previous_accuracy = eval_accuracy
        
        # if epoch % 100 == 0:
        #     print(f'save epoch_{int(epoch/100)}_2.pth')
        #     torch.save({'epoch': epoch, 
        #             'model_state_dict': model.state_dict(), 
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'loss': losses,}, f'{args.ckpt_dir}/epoch_{int(epoch/100)}_2.pth')
    
    torch.save({'epoch': epoch, 
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': losses,}, F'{args.ckpt_dir}/slot_final.pth')
           

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=512)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=350)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
