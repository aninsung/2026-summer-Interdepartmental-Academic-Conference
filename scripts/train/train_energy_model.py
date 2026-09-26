"""Train a patch energy model on train patients only.

The model learns local disagreement between perturbed masks and GT.  At test
it receives image/probability/mask channels only; GT never enters inference.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, sobel
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.brats2020_dataset import BraTS2020Dataset
from src.data.patient_split import load_or_create_patient_split
from src.models.energy_model import EnergyModel
from src.utils.refinement_inputs import energy_input


def make_examples(images, gts, seed=42, per_slice=3):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for image, gt in zip(images, gts):
        gt = gt > .5
        image2 = image.mean(axis=0) if image.ndim == 3 else image
        edge = np.hypot(sobel(image2, 0), sobel(image2, 1))
        edge /= max(float(edge.max()), 1e-6)
        for _ in range(per_slice):
            mask = gt.copy()
            op = int(rng.integers(0, 4))
            if op == 0:
                mask = binary_erosion(mask, iterations=int(rng.integers(1, 4)))
            elif op == 1:
                mask = binary_dilation(mask, iterations=int(rng.integers(1, 4)))
            elif op == 2:
                yy, xx = rng.integers(0, mask.shape[0], 2), rng.integers(0, mask.shape[1], 2)
                mask[max(0, yy[0]-3):yy[0]+3, max(0, xx[0]-3):xx[0]+3] ^= True
            noise = rng.normal(0, .04, mask.shape).astype(np.float32)
            probability = np.clip(mask.astype(np.float32) * .75 + (1-mask.astype(np.float32))*.15 + noise, .01, .99)
            x = energy_input(image, probability, mask.astype(np.float32))
            error = gaussian_filter((mask != gt).astype(np.float32), sigma=1.)
            boundary = binary_dilation(mask) ^ binary_erosion(mask)
            target = np.clip(.7 * error + .3 * (1-edge) * boundary, 0, 1).astype(np.float32)
            xs.append(x); ys.append(target[None])
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train_root', default='src/data/archive')
    ap.add_argument('--patient_split', default='checkpoints/patient_split.json')
    ap.add_argument('--max_patients', type=int, default=210)
    ap.add_argument('--modality', default='t1ce+flair')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num_workers', type=int, default=8,
                    help='CPU threads for concurrent MRI modality loading')
    ap.add_argument('--output', default='checkpoints/energy_model.pt')
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    split = load_or_create_patient_split(args.train_root, args.max_patients, args.patient_split)
    if not split.get('train'):
        raise ValueError('Training split is empty')
    ds = BraTS2020Dataset(args.train_root, modality=args.modality, target_size=128,
                          patient_ids=split['train'], simulate_rough=False, slice_selection='tumor',
                          num_workers=args.num_workers)
    images, gts, _ = ds.get_numpy_arrays()
    if len(images) == 0:
        raise ValueError('No training slices found')
    x, y = make_examples(images, gts, args.seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                        batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnergyModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = torch.nn.SmoothL1Loss()
    model.train()
    for epoch in range(args.epochs):
        total = 0.; count = 0
        for xb, yb in loader:
            pred = model(xb.to(device)); loss = loss_fn(pred, yb.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss) * len(xb); count += len(xb)
        print(f'epoch {epoch+1}/{args.epochs}: loss={total/max(count,1):.5f}')
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.cpu().state_dict(), 'schema_version': 1,
                'features': ['image','probability','mask','probability_mask'],
                'train_patients': sorted(split['train']), 'seed': args.seed,
                'target': 'perturbed-mask local error energy'}, out)
    print('saved', out)

if __name__ == '__main__':
    main()
