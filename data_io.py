import argparse
import csv
import io
import os
import zipfile
from PIL import Image
import numpy as np


def _write_zip(images, ids, zip_path, prefix):
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for img, i in zip(images, ids):
            # Match expected naming: <ID>_dummy.JPG (upper-case extension)
            name = f"{i:05d}_dummy.JPG"
            arcname = f"{prefix}/{name}"  # keep a folder prefix similar to real zips
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=95)
            z.writestr(arcname, buf.getvalue())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='dataset')
    ap.add_argument('--n_train', type=int, default=100)
    ap.add_argument('--n_val', type=int, default=20)
    ap.add_argument('--n_test', type=int, default=20)
    ap.add_argument('--image_size', type=int, default=256)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    total = args.n_train + args.n_val + args.n_test
    # IDs are 0..total-1 so score.csv row index matches ID (0-based)
    ids = np.arange(total)

    # Create random images and a synthetic continuous label
    images = []
    labels = []
    for i in ids:
        arr = rng.integers(0, 256, size=(args.image_size, args.image_size, 3), dtype=np.uint8)
        images.append(Image.fromarray(arr, mode='RGB'))
        # continuous label in [0,1)
        labels.append(float(rng.random()))

    # Write score.csv with one column matching original access [row][0]
    os.makedirs(args.out_dir, exist_ok=True)
    score_path = os.path.join(args.out_dir, 'score.csv')
    with open(score_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        for y in labels:
            w.writerow([y])

    # Split and write ZIPs
    train_imgs = images[:args.n_train]
    train_ids = ids[:args.n_train]
    val_imgs = images[args.n_train:args.n_train + args.n_val]
    val_ids = ids[args.n_train:args.n_train + args.n_val]
    test_imgs = images[args.n_train + args.n_val:]
    test_ids = ids[args.n_train + args.n_val:]

    _write_zip(train_imgs, train_ids, os.path.join(args.out_dir, 'ALC100train.zip'), 'ALC100train')
    _write_zip(val_imgs, val_ids, os.path.join(args.out_dir, 'ALC100val.zip'), 'ALC100val')
    _write_zip(test_imgs, test_ids, os.path.join(args.out_dir, 'ALC100test.zip'), 'ALC100test')

    print('Dummy dataset created under:', args.out_dir)
    print(' - score.csv')
    print(' - ALC100train.zip / ALC100val.zip / ALC100test.zip')


if __name__ == '__main__':
    main()
