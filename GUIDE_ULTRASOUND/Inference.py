import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from models.GUIDE.GUIDE import GUIDE
from torch.nn import functional as F
import csv
from tqdm import tqdm
from Dataset import Dataset_nogaze

def infer_e2e():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True

    size = 224
    device = torch.device('cuda:0')
    test_dir = 'Data'
    csv_path = 'Data/image_names.csv'
    model = GUIDE(num_classes=2)
    output_gaze_dir = "Pred_Gaze"
    model_path = 'output/weights/checkpoint_final.pth'
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval().to(device)

    dataset_test = Dataset_nogaze(test_dir, csv_path, 'test', size)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)

    All_Infor = []
    with torch.no_grad():
        for img,  label, img_path in tqdm(dataloader_test):
            img = img.to(device)
            label = label.to(device)

            output = model(img)
            evidences = F.softplus(output)
            alpha = evidences + 1
            S = torch.sum(alpha, dim=1, keepdim=True)
            u = 2 / S
            gaze_pred = model.get_gaze()
            _, pred = torch.max(output, 1)
            pred_score = torch.nn.Softmax(dim=1)(output)
            All_Infor.append([img_path[0].split('/')[-1],
                              u[0].cpu().detach().numpy().item(),
                              label[0].cpu().detach().numpy(),
                              pred[0].cpu().detach().numpy(),
                              pred_score[0][1].cpu().detach().numpy()])

            gaze_pred = (gaze_pred - gaze_pred.min()) / (gaze_pred.max() - gaze_pred.min())
            gaze_pred = np.array(gaze_pred[0][0].detach().cpu()) * 255
            output_img = Image.fromarray(gaze_pred).convert('L')
            name = img_path[0].split('/')[-1]
            output_img.save(output_gaze_dir + name)

        Results_Heads = ["Imagefiles", 'uncertainty', 'label', 'pred']
        pred_results = "results.csv"
        with open(pred_results, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(Results_Heads)
            writer.writerows(All_Infor)

if __name__ == '__main__':
    infer_e2e()
