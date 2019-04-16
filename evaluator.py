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
        max_e = self.Eval_Emeasure()
        print('{} dataset with {} method get {:.4f} mae, {:.4f} max-fmeasure, {:.4f} max-Emeasure.'.format(self.dataset, self.method, mae, max_f, max_e))
        self.LOG('{} dataset with {} method get {:.4f} mae, {:.4f} max-fmeasure, {:.4f} max-Emeasure.'.format(self.dataset, self.method, mae, max_f, max_e))
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
    def Eval_Emeasure(self):
        print('eval[EMeasure]:{} dataset with {} method.'.format(self.dataset, self.method))
        avg_e, img_num = 0.0, 0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt in self.loader:
                pred = trans(pred)
                gt = trans(gt)
                max_e = self._eval_e(pred, gt, 255)
                if max_e == max_e:
                    avg_e += max_e
                    img_num += 1.0
                
            avg_e /= img_num
            return avg_e

    def LOG(self, output):
        with open(self.logfile, 'a') as f:
            f.write(output)

    def _eval_e(self, y_pred, y, num):
        score = torch.zeros(num)
        for i in range(num):
            fm = y_pred - y_pred.mean()
            gt = y - y.mean()
            align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
            enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
            score[i] = torch.sum(enhanced) / (y.numel() - 1 + 1e-20)
        return score.max()

    def _eval_pr(self, y_pred, y, num):
        prec, recall = torch.zeros(num), torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
        return prec, recall