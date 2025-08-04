"""
ablation of lambda
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .encoder import Res101Encoder





class FewShotSeg(nn.Module):

    def __init__(self, pretrained_weights="deeplabv3"):
        super().__init__()

        # Encoder
        self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
                                     pretrained_weights=pretrained_weights)  # or "resnet101"
        self.device = torch.device('cuda')
        self.scaler = 20.0
        self.criterion = nn.NLLLoss()
        self.criterion_MSE = nn.MSELoss()
        self.alpha = torch.Tensor([1.0, 0.])


    def forward(self, supp_imgs, supp_mask, qry_imgs, qry_mask=None, train=False, t_loss_scaler=1, n_iters=30):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
            qry_mask: query mask
                1 x H x W  tensor
        """

        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = len(qry_imgs)
        assert self.n_ways == 1  # for now only one-way, because not every shot has multiple sub-images
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]
        supp_bs = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)  # B x Wa x Sh x H x W
        # supp_mask: (1, 1, 1, 256, 256)

        # Dilate the mask
        kernel = np.ones((10, 10), np.uint8)
        supp_mask_ = supp_mask.cpu().numpy()[0][0][0]
        supp_dilated_mask = cv2.dilate(supp_mask_, kernel, iterations=1)  # (256, 256)
        supp_periphery_mask = supp_dilated_mask - supp_mask_
        supp_periphery_mask = np.reshape(supp_periphery_mask, (supp_bs, self.n_ways, self.n_shots,
                                                               np.shape(supp_periphery_mask)[0],
                                                               np.shape(supp_periphery_mask)[1]))
        supp_dilated_mask = np.reshape(supp_dilated_mask, (supp_bs, self.n_ways, self.n_shots,
                                                           np.shape(supp_dilated_mask)[0],
                                                           np.shape(supp_dilated_mask)[1]))
        supp_periphery_mask = torch.tensor(supp_periphery_mask).cuda()  # (1, 1, 1, 256, 256)  B x Wa x Sh x H x W
        supp_dilated_mask = torch.tensor(supp_dilated_mask).cuda()  # (1, 1, 1, 256, 256)  B x Wa x Sh x H x W

        # Extract features #
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)
        img_fts, tao = self.encoder(imgs_concat)
        supp_fts = [img_fts[dic][:self.n_ways * self.n_shots * supp_bs].view(  # B x Wa x Sh x C x H' x W'
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]
        qry_fts = [img_fts[dic][self.n_ways * self.n_shots * supp_bs:].view(  # B x N x C x H' x W'
            qry_bs, self.n_queries, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]

        # Get threshold #
        self.t = tao[self.n_ways * self.n_shots * supp_bs:]  # t for query features
        self.thresh_pred = [self.t for _ in range(self.n_ways)]

        self.t_ = tao[:self.n_ways * self.n_shots * supp_bs]  # t for support features
        self.thresh_pred_ = [self.t_ for _ in range(self.n_ways)]

        # Compute loss #
        align_loss = torch.zeros(1).to(self.device)
        mse_loss = torch.zeros(1).to(self.device)
        loss_qry = torch.zeros(1).to(self.device)
        outputs = []

        H, W = qry_fts[0].shape[-2:]
        supp_mask_size=supp_mask.shape[-2:]

        mask = F.interpolate(supp_mask.view(self.n_ways*self.n_shots*supp_bs,1,supp_mask_size[0],supp_mask_size[1]), size=(H, W), mode='bilinear', align_corners=True)
        mask=mask.view(supp_bs,self.n_ways,self.n_shots,H,W)

        for epi in range(supp_bs):
            # Extract prototypes #
            # First, object region prototypes #
            supp_fts_ = [[[self.getFeatures(supp_fts[n][[epi], way, shot], supp_mask[[epi], way, shot])
                           for shot in range(self.n_shots)] for way in range(self.n_ways)] for n in
                         range(len(supp_fts))]
            fg_prototypes = [self.getPrototype(supp_fts_[n]) for n in range(len(supp_fts))]  # prototype for support n*way[1*C],没有shot,因为做平均了


            # QPC module
            qry_pred_1 = [torch.stack(
                [self.getPred(qry_fts[n][epi], fg_prototypes[n][way], self.thresh_pred[way])
                 for way in range(self.n_ways)], dim=1) for n in range(len(qry_fts))]  # N x Wa x H' x W'

            if train:
                # qry_pred = [torch.stack(
                #     [self.getPred(qry_fts[n][epi], fg_prototypes[n][way], self.thresh_pred[way])
                #      for way in range(self.n_ways)], dim=1) for n in range(len(qry_fts))]  # N x Wa x H' x W'
                # qry_prototype_coarse = [self.getFeatures(qry_fts[n][epi], qry_pred[n][epi]) for n in
                #                         range(len(qry_fts))]
                qry_pred = [F.interpolate(qry_pred_1[n], size=img_size, mode='bilinear', align_corners=True)
                            for n in range(len(qry_fts))]
                preds = [self.alpha[n] * qry_pred[n] for n in range(len(qry_fts))]
                preds = torch.sum(torch.stack(preds, dim=0), dim=0) / torch.sum(self.alpha)
                preds = torch.cat((1.0 - preds, preds), dim=1)

                qry_label = torch.full_like(qry_mask[epi], 255, device=qry_mask.device)
                qry_label[qry_mask[epi] == 1] = 1
                qry_label[qry_mask[epi] == 0] = 0
                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(preds, eps, 1 - eps))
                loss_qry += self.criterion(log_prob, qry_label[None, ...].long()) / self.n_shots / self.n_ways
            #print(f"qry_pred_1[0]:{qry_pred_1[0].shape}")  N x Wa x H' x W',N是查询图片数量

            qry_mask=[torch.stack((1.0 - qry_pred_1[n], qry_pred_1[n]), dim=2).max(2)[1] for n in range(len(qry_pred_1))]
            #print(f"qry_mask:{qry_mask[0].shape}") N x Wa x H' x W'

            qry_prototype_coarse = [self.getFeatures(qry_fts[n][epi], qry_mask[n][epi]) for n in range(len(qry_fts))]
            #print(f"qry_prototype_coarse:{qry_prototype_coarse[0].shape}") n*[1*C]

            FP=[0.5*fg_prototypes[n][epi]+0.5*qry_prototype_coarse[n] for n in range(len(qry_fts))]
            #print(f"FP:{FP[0].shape}") n*[1*C]

            qry_pred_2 = [self.getPred(qry_fts[n][epi], FP[n], self.thresh_pred[epi]) for n in range(len(qry_fts))]  # N x Wa x H' x W'
            #print(f"qry_pred_2:{qry_pred_2[0].shape}") 1*64*64

            qry_pred = [F.interpolate(qry_pred_2[n][None, ...], size=img_size, mode='bilinear', align_corners=True)
                        for n in range(len(qry_fts))]
            preds = [self.alpha[n] * qry_pred[n] for n in range(len(qry_fts))]
            preds = torch.sum(torch.stack(preds, dim=0), dim=0) / torch.sum(self.alpha)
            preds = torch.cat((1.0 - preds, preds), dim=1)      #1*2*256*256

            qry_mask=preds.max(1)[1]

            outputs.append(preds)

            #qry_mask = [torch.stack((1.0 - qry_pred_2[n], qry_pred_2[n]), dim=1).max(1)[1] for n in range(len(qry_pred_2))]
            #print(f"qry_mask:{qry_mask[0].shape}") 1*64*64
            # Prototype alignment loss #
            if train:
                align_loss_epi = self.alignLoss([supp_fts[n][epi] for n in range(len(supp_fts))],
                                                [qry_fts[n][epi] for n in range(len(qry_fts))],
                                                preds, supp_mask[epi])
                align_loss += align_loss_epi

            # Mse alignment loss #
            if train:
                mse_loss_epi = self.proto_alignLoss([supp_fts[n][epi] for n in range(len(supp_fts))],
                                                    [qry_fts[n][epi] for n in range(len(qry_fts))],
                                                    preds, supp_mask[epi], fg_prototypes)
                mse_loss += mse_loss_epi

        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])

        #return 0
        return output,align_loss / supp_bs, mse_loss / supp_bs, loss_qry / supp_bs

    def getPred(self, fts, prototype, thresh):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """

        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))


        return pred

    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """

        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')

        # masked fg features
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C

        return masked_fts

    def getPrototype(self, fg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]  ## concat all fg_fts

        return fg_prototypes

    def alignLoss(self, supp_fts, qry_fts, pred, fore_mask):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)  # N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                # Get prototypes
                qry_fts_ = [[self.getFeatures(qry_fts[n], pred_mask[way + 1])] for n in range(len(qry_fts))]
                fg_prototypes = [self.getPrototype([qry_fts_[n]]) for n in range(len(supp_fts))]

                # Get predictions
                supp_pred = [self.getPred(supp_fts[n][way, [shot]], fg_prototypes[n][way], self.thresh_pred_[way])
                             for n in range(len(supp_fts))]  # N x Wa x H' x W'
                supp_pred = [F.interpolate(supp_pred[n][None, ...], size=fore_mask.shape[-2:], mode='bilinear',
                                           align_corners=True)
                             for n in range(len(supp_fts))]

                # Combine predictions of different feature maps
                preds = [self.alpha[n] * supp_pred[n] for n in range(len(supp_fts))]
                preds = torch.sum(torch.stack(preds, dim=0), dim=0) / torch.sum(self.alpha)

                pred_ups = torch.cat((1.0 - preds, preds), dim=1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))



                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss

    def proto_alignLoss(self, supp_fts, qry_fts, pred, fore_mask, supp_prototypes):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)  # N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'

        # Compute the support loss
        loss_sim = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                # Get prototypes
                qry_fts_ = [[self.getFeatures(qry_fts[n], pred_mask[way + 1])] for n in range(len(qry_fts))]
                fg_prototypes = [self.getPrototype([qry_fts_[n]]) for n in range(len(supp_fts))]

                # Combine prototypes from different scales
                fg_prototypes = [self.alpha[n] * fg_prototypes[n][way] for n in range(len(supp_fts))]
                fg_prototypes = torch.sum(torch.stack(fg_prototypes, dim=0), dim=0) / torch.sum(self.alpha)
                supp_prototypes_ = [self.alpha[n] * supp_prototypes[n][way] for n in range(len(supp_fts))]
                supp_prototypes_ = torch.sum(torch.stack(supp_prototypes_, dim=0), dim=0) / torch.sum(self.alpha)

                # Compute the MSE loss
                loss_sim += self.criterion_MSE(fg_prototypes, supp_prototypes_)

        return loss_sim

    def pr_transformer(self, x, init_mask, s_x, supp_mask):
        """
        x:1*C*H*W
        init_mask:1*H*W
        s_x:1*C*H*W
        supp_mask:1*H*W
        """


        return





