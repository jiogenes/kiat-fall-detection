import torch, einops, pathlib, cv2, json, shutil
import numpy as np
import matplotlib.pylab as plt
from matplotlib import patches
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from custom_models.custom_models import FallDetector
from custom_models.custom_data import URDataset


EPOCHS = 30
DEVICE = 0
LR = 1e-3


if __name__ == "__main__":
    model = FallDetector()
    model = model.to(DEVICE)
    
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.5, 0.9))
    dataset = URDataset("output/UR_fall_detection2")
    dloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for e in range(EPOCHS):
        accs = []
        gts = []
        preds = []
        for frames, labels in dloader:
            frames = frames.to(0)
            labels = labels.to(0).float()

            pred = model(frames)
            loss = criterion(pred, labels)

            cat_pred = (pred >= 0.5).float()
            acc = (labels == cat_pred).float().mean()
            accs.append(acc)
            preds.append(cat_pred.detach().cpu().numpy())
            gts.append(labels.cpu().numpy())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        gts = np.concatenate(gts)
        preds = np.concatenate(preds)
        f1 = f1_score(gts, preds)
        acc = sum(accs) / len(accs)
        print(f"Acc: {acc:.4f}, F1: {f1:.4f}")

        if f1 > 0.9 and acc > 0.9: break

    torch.save(model.state_dict(), "./fall_detector.pt")
    shutil.copy("models/fall_detector.pt", "falldetection_openpifpaf_custom/fall_detector.pt")

