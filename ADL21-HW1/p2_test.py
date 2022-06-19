import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch, csv, random
from tqdm import trange
import numpy as np

from dataset import TaggingDataset_test
from utils import Vocab
from model import TaggingClassifier, SlotRNN

# ./slot_tag.sh data/slot/test.json pred_slot.csv

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

    data = json.loads(args.test_file.read_text())
    dataset = TaggingDataset_test(data, vocab, tag2idx)

    test_data_loader = torch.utils.data.DataLoader(dataset,
                                        batch_size = args.batch_size,
                                        shuffle=False,
                                        num_workers=0,
                                        collate_fn=dataset.collate_fn,
                                        )

    embeddings = torch.load(args.cache_dir / "slot_embeddings.pt")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_class = dataset.num_classes + 1
    # model = TaggingClassifier(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, num_class, dataset.max_len)

    model = SlotRNN(embeddings, args.hidden_size, num_class, args.bidirectional)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=9).to(device)
    criterion.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    csvfile = open(args.pred_file, 'w', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'tags']) 

    model.eval()
    idx = 0
    with torch.no_grad():

        for i, (targets, length) in enumerate(test_data_loader):
            targets = targets.to(device)

            outputs = model(targets).float()

            pred = torch.argmax(outputs, dim=2).reshape(-1, dataset.max_len)

            pred = pred.cpu().numpy()

            for pred_id, le in zip(pred, length):
                intent = [dataset.idx2label(label.item()) for label in pred_id[:le]]
                intent = ' '.join(intent)
                row = [f'test-{idx}', intent]
                writer.writerow(row)
                idx += 1
    
    print('done')

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/test.json",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Directory to save the model file.",
        default="slot_latest.pth",
    )
    parser.add_argument("--pred_file", type=Path, default="pred_slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-4)

    # data loader
    parser.add_argument("--batch_size", type=int, default=512)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=500)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
