import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch, csv, random
import numpy as np

from dataset import SeqClsDataset_test
from model import SeqClassifier
from utils import Vocab

# ./intent_cls.sh data/intent/test.json pred_intent.csv

seed = 567
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

def main(args):
    with open(args.cache_dir / "intent_vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset_test(data, vocab, intent2idx, args.max_len)

    test_data_loader = torch.utils.data.DataLoader(dataset,
                                        batch_size = args.batch_size,
                                        shuffle=False,
                                        num_workers=0,
                                        collate_fn=dataset.collate_fn,
                                        )
    embeddings = torch.load(args.cache_dir / "intent_embeddings.pt")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_class = dataset.num_classes
    model = SeqClassifier(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, num_class)
    model.to(device)

    criterion = torch.nn.BCELoss().to(device)
    criterion.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    csvfile = open(args.pred_file, 'w', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'intent']) 

    model.eval()
    correct = 0
    idx = 0
    all_pred = []
    with torch.no_grad():
        for i, (targets, length) in enumerate(test_data_loader):
            targets = targets.to(device)

            outputs = model(targets, length).float()

            pred = torch.argmax(outputs, dim=1)

            for pred_id in pred:
                intent = dataset.idx2label(pred_id.item())
                row = [f'test-{idx}', intent]
                writer.writerow(row)
                idx += 1

    print('done')

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/intent/test.json",
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
        help="Path to model checkpoint.",
        default="intent_latest.pth",
    )
    parser.add_argument("--pred_file", type=Path, default="pred_intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=512)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)