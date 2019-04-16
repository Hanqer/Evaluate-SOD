import threading
import torch
import os
from torchvision import transforms
class Eval_thread(threading.Thread):
    def __init__(self, loader, method, dataset, output_dir):
        threading.Thread.__init__(self)
        self.loader = loader
        self.method = method
        self.dataset = dataset
        self.logfile = os.path.join(output_dir, 'result.txt')
    def run(self):
        mae = self.Eval_mae()
        max_f = self.Eval_fmeasure()
        print('{} dataset with {} method get {:.4f} mae, {:.4f} max-fmeasure.'.format(self.dataset, self.method, mae, max_f))
        self.LOG('{} dataset with {} method get {:.4f} mae, {:.4f} max-fmeasure.'.format(self.dataset, self.method, mae, max_f))
    def Eval_mae(self):
        print('eval[MAE]:{} dataset with {} method.'.format(self.dataset, self.method))
        avg_mae, img_num = 0.0, 0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt in self.loader:
                pred = trans(pred)
                gt = trans(gt)
                mea = torch.abs(pred - gt).mean()
                if mea == mea: # for Nan
                    avg_mae += mea
                    img_num += 1.0
            avg_mae /= img_num
            return avg_mae.item()
    
    def Eval_fmeasure(self):
        print('eval[FMeasure]:{} dataset with {} method.'.format(self.dataset, self.method))
        beta2 = 0.3
        avg_p, avg_r, img_num = 0.0, 0.0, 0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt in self.loader:
                pred = trans(pred)
                gt = trans(gt)
                prec, recall = self._eval_pr(pred, gt, 255)
                avg_p += prec
                avg_r += recall
                img_num += 1.0
            avg_p /= img_num
            avg_r /= img_num
            score = (1 + beta2) * avg_p * avg_r / (beta2 * avg_p + avg_r)
            score[score != score] = 0 # for Nan
            
            return score.max().item()

    def LOG(self, output):
        with open(self.logfile, 'a') as f:
            f.write(output)

    def _eval_pr(self, y_pred, y, num):
        prec, recall = torch.zeros(num), torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
        return prec, recall