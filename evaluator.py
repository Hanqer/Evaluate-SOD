import threading
import torch
import numpy as np
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
        s = self.Eval_Smeasure()
        
        print('{} dataset with {} method get {:.4f} mae, {:.4f} max-fmeasure, {:.4f} max-Emeasure, {:.4f} S-measure..'.format(self.dataset, self.method, mae, max_f, max_e, s))
        # self.LOG('{} dataset with {} method get {:.4f} mae, {:.4f} max-fmeasure, {:.4f} max-Emeasure.\n'.format(self.dataset, self.method, mae, max_f, max_e))
    def Eval_mae(self):
        print('eval[MAE]:{} dataset with {} method.'.format(self.dataset, self.method))
        avg_mae, img_num = 0.0, 0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt in self.loader:
                pred = trans(pred).cuda()
                gt = trans(gt).cuda()
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
                pred = trans(pred).cuda()
                gt = trans(gt).cuda()
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
                pred = trans(pred).cuda()
                gt = trans(gt).cuda()
                max_e = self._eval_e(pred, gt, 255)
                if max_e == max_e:
                    avg_e += max_e
                    img_num += 1.0
                
            avg_e /= img_num
            return avg_e
    def Eval_Smeasure(self):
        print('eval[SMeasure]:{} dataset with {} method.'.format(self.dataset, self.method))
        alpha, avg_q, img_num = 0.5, 0.0, 0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt in self.loader:
                pred = trans(pred).cuda()
                gt = trans(gt).cuda()
                Q = alpha * self._S_object(pred, gt) + (1-alpha) * self._S_region(pred, gt)
                img_num += 1.0
                avg_q += Q.item()
            avg_q /= img_num
            return avg_q
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
        thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
        return prec, recall
    
    def _S_object(self, pred, gt):
        fg = torch.where(gt==0, torch.zeros_like(pred), pred)
        bg = torch.where(gt==1, torch.zeros_like(pred), 1-pred)
        o_fg = self._object(fg, gt)
        o_bg = self._object(bg, gt)
        u = gt.mean()
        Q = u * o_fg + (1-u) * o_bg
        return Q

    def _object(self, pred, gt):
        temp = pred[gt == 0]
        x = temp.mean()
        sigma_x = temp.std()
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)
        
        return score

    def _S_region(self, pred, gt):
        X, Y = self._centroid(gt)
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self._divideGT(gt, X, Y)
        p1, p2, p3, p4 = self._dividePrediction(pred, X, Y)
        Q1 = self._ssim(p1, gt1)
        Q2 = self._ssim(p2, gt2)
        Q3 = self._ssim(p3, gt3)
        Q4 = self._ssim(p4, gt4)
        Q = w1*Q1 + w2*Q2 + w3*Q3 + w4*Q4
        return Q
    
    def _centroid(self, gt):
        rows, cols = gt.size()[-2:]
        gt = gt.view(rows, cols)
        if gt.sum() == 0:
            X = torch.eye(1).cuda() * (cols // 2)
            Y = torch.eye(1).cuda() * (rows // 2)
        else:
            total = gt.sum()
            i = torch.from_numpy(np.arange(0,cols)).cuda().float()
            j = torch.from_numpy(np.arange(0,rows)).cuda().float()
            X = (gt.sum(dim=0)*i).sum() // total
            Y = (gt.sum(dim=1)*j).sum() // total
        return X.long(), Y.long()
    
    def _divideGT(self, gt, X, Y):
        h, w = gt.size()[-2:]
        area = h*w
        gt = gt.view(h, w)
        LT = gt[:Y, :X]
        RT = gt[:Y, X+1:w]
        LB = gt[Y+1:h, :X]
        RB = gt[Y+1:h, X+1:w]
        w1 = X * Y / area
        w2 = (w - X) * Y / area
        w3 = X * (h - Y) / area
        w4 = 1 - w1 - w2 - w3
        return LT, RT, LB, RB, w1, w2, w3, w4

    def _dividePrediction(self, pred, X, Y):
        h, w = pred.size()[-2:]
        pred = pred.view(h, w)
        LT = pred[:Y, :X]
        RT = pred[:Y, X+1:w]
        LB = pred[Y+1:h, :X]
        RB = pred[Y+1:h, X+1:w]
        return LT, RT, LB, RB

    def _ssim(self, pred, gt):
        gt = gt.float()
        h, w = pred.size()[-2:]
        N = h*w
        x = pred.mean()
        y = gt.mean()
        sigma_x2 = ((pred - x)*(pred - x)).sum() / (N - 1 + 1e-20)
        sigma_y2 = ((gt - y)*(gt - y)).sum() / (N - 1 + 1e-20)
        sigma_xy = ((pred - x)*(gt - y)).sum() / (N - 1 + 1e-20)
        
        aplha = 4 * x * y *sigma_xy
        beta = (x*x + y*y) * (sigma_x2 + sigma_y2)

        if aplha != 0:
            Q = aplha / (beta + 1e-20)
        elif aplha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0
        return Q