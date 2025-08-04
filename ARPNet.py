"""
ablation of lambda
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .encoder import Res101Encoder

from models.transformer import CyCTransformer as PRTransformer
from models.ops.modules import MSDeformAttn

from models.seed_init import place_seed_points




class FewShotSeg(nn.Module):

    def __init__(self, pretrained_weights="deeplabv3"):
        super().__init__()

        # Encoder
        self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
                                     pretrained_weights=pretrained_weights)  # or "resnet101"
        self.device = torch.device('cuda')
        self.scaler = 20.0
        self.fg_sampler = np.random.RandomState(1289)
        self.fg_num = 64  # number of foreground partitions
        self.criterion = nn.NLLLoss()
        self.criterion_MSE = nn.MSELoss()
        self.alpha = torch.Tensor([1.0, 0.])

        reduce_dim=512

        self.transformer = PRTransformer(embed_dims=reduce_dim, num_points=9)

        self.qry_merge_feat = nn.Sequential(
            nn.Conv2d(reduce_dim * 2, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.supp_merge_feat = nn.Sequential(
            nn.Conv2d(reduce_dim * 2, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )


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
        loss_qry1 = torch.zeros(1).to(self.device)
        loss_qry2 = torch.zeros(1).to(self.device)
        outputs = []

        H, W = qry_fts[0].shape[-2:]

        for epi in range(supp_bs):
            # Extract prototypes #
            # First, object region prototypes #
            supp_fts_ = [[[self.getFeatures(supp_fts[n][[epi], way, shot], supp_mask[[epi], way, shot])
                           for shot in range(self.n_shots)] for way in range(self.n_ways)] for n in
                         range(len(supp_fts))]
            fg_prototypes = [self.getPrototype(supp_fts_[n]) for n in range(len(supp_fts))]  # prototype for support n*way[1*C],没有shot,因为做平均了

            # perform a new prediction operation
            fg_partition_prototypes = [self.compute_multiple_prototypes(
                self.fg_num, supp_fts[epi][[epi], epi, shot], supp_mask[[epi], epi, shot], self.fg_sampler).squeeze(0).unsqueeze(-1).unsqueeze(-1)
                for shot in range(self.n_shots)]

            #print(f"fg_partition_prototypes:{fg_partition_prototypes[0].shape}")


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
                loss_qry1 += self.criterion(log_prob, qry_label[None, ...].long()) / self.n_shots / self.n_ways
            #print(f"qry_pred_1[0]:{qry_pred_1[0].shape}")  N x Wa x H' x W',N是查询图片数量

            preds=[torch.stack((1.0 - qry_pred_1[n], qry_pred_1[n]), dim=2).max(2)[1] for n in range(len(qry_pred_1))]
            #print(f"qry_mask:{qry_mask[0].shape}") N x Wa x H' x W'

            qry_prototype_coarse = [self.getFeatures(qry_fts[n][epi], preds[n][epi]) for n in range(len(qry_fts))]
            #print(f"qry_prototype_coarse:{qry_prototype_coarse[0].shape}") n*[1*C]

            FP=[0.5*fg_prototypes[n][epi]+0.5*qry_prototype_coarse[n] for n in range(len(qry_fts))]
            #print(f"FP:{FP[0].shape}") n*[1*C]

            qry_pred_2 = [self.getPred(qry_fts[n][epi], FP[n], self.thresh_pred[epi]) for n in range(len(qry_fts))]
            #print(f"qry_pred_2:{qry_pred_2[0].shape}") 1*64*64

########################################
            qry_pred = [F.interpolate(qry_pred_2[n][None, ...], size=img_size, mode='bilinear', align_corners=True)
                        for n in range(len(qry_fts))]
            preds = [self.alpha[n] * qry_pred[n] for n in range(len(qry_fts))]
            preds = torch.sum(torch.stack(preds, dim=0), dim=0) / torch.sum(self.alpha)
            preds = torch.cat((1.0 - preds, preds), dim=1)

            Pre_pred = preds.clone()
#########################################

            if train:
                # qry_pred = [torch.stack(
                #     [self.getPred(qry_fts[n][epi], fg_prototypes[n][way], self.thresh_pred[way])
                #      for way in range(self.n_ways)], dim=1) for n in range(len(qry_fts))]  # N x Wa x H' x W'
                # qry_prototype_coarse = [self.getFeatures(qry_fts[n][epi], qry_pred[n][epi]) for n in
                #                         range(len(qry_fts))]
                qry_pred = [F.interpolate(qry_pred_2[n][None, ...], size=img_size, mode='bilinear', align_corners=True)
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
                loss_qry2 += self.criterion(log_prob, qry_label[None, ...].long()) / self.n_shots / self.n_ways

            qry_pred = [F.interpolate(qry_pred_2[n][None, ...], size=(H,W), mode='bilinear', align_corners=True)
                        for n in range(len(qry_fts))]
            preds = [self.alpha[n] * qry_pred[n] for n in range(len(qry_fts))]
            preds = torch.sum(torch.stack(preds, dim=0), dim=0) / torch.sum(self.alpha)
            preds = torch.cat((1.0 - preds, preds), dim=1)      #1*2*256*256

            preds=preds.max(1)[1]

            #print(f"qry_fts[0][epi]:{qry_fts[0][epi].shape},FP[0]:{FP[0].shape},qry_pred_2[1]:{qry_pred_2[1].shape}")
            #qry_fts[0][epi]:torch.Size([1, 512, 64, 64]),FP[0]:torch.Size([1, 512]),qry_pred_2[1]:torch.Size([1, 32, 32])
            query_cat_feat = [qry_fts[0][epi], FP[0].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)]
            query_feat = self.qry_merge_feat(torch.cat(query_cat_feat, dim=1))

            #print(f"supp_fts[0][epi]:{supp_fts[0][epi][epi].shape}")   1*512*64*64
            to_merge_fts = [supp_fts[0][epi][epi], FP[0].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)]
            aug_supp_feat = torch.cat(to_merge_fts, dim=1)
            aug_supp_feat = self.supp_merge_feat(aug_supp_feat)


            qry_label = torch.full_like(qry_mask[epi], 255, device=qry_mask.device)
            qry_label[qry_mask[epi] == 1] = 1
            qry_label[qry_mask[epi] == 0] = 0

            #print(f"query_feat:{query_feat.shape},qry_label:{qry_label.shape},aug_supp_feat:{aug_supp_feat.shape},mask[epi]:{mask[epi].shape},FP[0]:{FP[0].shape},preds:{preds.shape}")
            #query_feat:torch.Size([1, 512, 64, 64]),qry_label:torch.Size([256, 256]),aug_supp_feat:torch.Size([1, 512, 64, 64]),mask[epi]:torch.Size([1, 1, 64, 64]),FP[0]:torch.Size([1, 512]),preds:torch.Size([1, 64, 64])
            pr_prototypes=self.transformer(query_feat,qry_label.unsqueeze(0).float(),aug_supp_feat,supp_mask[epi].clone().float(),fg_partition_prototypes[0],preds.detach(),self.thresh_pred[epi])
            #print(f"query_feat_list:{query_feat_list.shape},pr_prototypes:{pr_prototypes.shape},qry_outputs_mask_list[0]:{qry_outputs_mask_list[0].shape}")
            #query_feat_list:torch.Size([1, 512, 64, 64]),pr_prototypes:torch.Size([1, 512]),qry_outputs_mask_list[0]:torch.Size([1, 2, 64, 64])
            #print(f"pr_prototypes:{pr_prototypes.shape}")
            query_feat=qry_fts[epi][epi]
            qry_pred=self.getPred(query_feat,pr_prototypes,self.thresh_pred[epi]).max(0)[0]

            #print(f"qry_pred:{qry_pred.shape}")    1*64*64
            preds=F.interpolate(qry_pred[None,None, ...], size=img_size, mode='bilinear', align_corners=True)
            preds=self.alpha[0]*preds
            preds = torch.cat((1.0 - preds, preds), dim=1)
            outputs.append(preds)

            # if train:
            #     for qy_id, qry_out in enumerate(qry_outputs_mask_list):
            #         qry_out = F.interpolate(qry_out, size=img_size, mode='bilinear', align_corners=True)
            #
            #         qry_label = torch.full_like(qry_mask[epi], 255, device=qry_mask.device)
            #         qry_label[qry_mask[epi] == 1] = 1
            #         qry_label[qry_mask[epi] == 0] = 0
            #         # Compute Loss
            #         eps = torch.finfo(torch.float32).eps
            #         log_prob = torch.log(torch.clamp(qry_out, eps, 1 - eps))
            #         mse_loss += self.criterion(log_prob, qry_label[None, ...].long()) / self.n_shots / self.n_ways
            #     mse_loss=mse_loss/len(qry_outputs_mask_list)

            # Prototype alignment loss #
            if train:
                align_loss_epi = self.alignLoss([supp_fts[n][epi] for n in range(len(supp_fts))],
                                                [qry_fts[n][epi] for n in range(len(qry_fts))],
                                                preds, supp_mask[epi])
                align_loss += align_loss_epi

                loss_qry=0.5*loss_qry1+0.5*loss_qry2

            if train:
                mse_loss_epi = self.proto_alignLoss([supp_fts[n][epi] for n in range(len(supp_fts))],
                                                    [qry_fts[n][epi] for n in range(len(qry_fts))],
                                                    preds, supp_mask[epi], fg_prototypes)
                mse_loss += mse_loss_epi

        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])

        #return 0
        return output,Pre_pred, align_loss / supp_bs, loss_qry / supp_bs, mse_loss / supp_bs

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

    def compute_multiple_prototypes(self, fg_num, sup_fts, sup_fg, sampler):
        """
        Modified from Jian-Wei Zhang

        Parameters
        ----------
        fg_num: int
            Foreground partition numbers
        sup_fts: torch.Tensor
             [B, C, h, w], float32
        sup_fg: torch. Tensor
             [B, h, w], float32 (0,1)
        sampler: np.random.RandomState

        Returns
        -------
        fg_proto: torch.Tensor
            [B, k, C], where k is the number of foreground proxies

        """

        B, C, h, w = sup_fts.shape  # B=1, C=512
        fg_mask_ = F.interpolate(sup_fg.unsqueeze(0), size=sup_fts.shape[-2:], mode='bilinear')
        fg_mask = fg_mask_.squeeze(0).bool()  # [B, h, w] --> bool
        batch_fg_protos = []

        for b in range(B):
            fg_protos = []

            fg_mask_i = fg_mask[b]  # [h, w]

            # Check if zero
            with torch.no_grad():
                if fg_mask_i.sum() < fg_num:
                    fg_mask_i = fg_mask[b].clone()  # don't change original mask
                    fg_mask_i.view(-1)[:fg_num] = True

            # Iteratively select farthest points as centers of foreground local regions
            #all_centers = []
            #first = True
            pts = torch.stack(torch.where(fg_mask_i), dim=1)    #N*2
            #print(f"pts:{pts.shape}")
            # for _ in range(fg_num):
            #     if first:
            #         i = sampler.choice(pts.shape[0])
            #         first = False
            #     else:
            #         dist = pts.reshape(-1, 1, 2) - torch.stack(all_centers, dim=0).reshape(1, -1, 2)
            #         # choose the farthest point
            #         i = torch.argmax((dist ** 2).sum(-1).min(1)[0])
            #     pt = pts[i]  # center y, x
            #     all_centers.append(pt)
            all_centers=place_seed_points(fg_mask_.squeeze(0).squeeze(0))

            # Assign fg labels for fg pixels
            dist = pts.reshape(-1, 1, 2) - all_centers.reshape(1, -1, 2)
            fg_labels = torch.argmin((dist ** 2).sum(-1), dim=1)

            # Compute fg prototypes
            fg_feats = sup_fts[b].permute(1, 2, 0)[fg_mask_i]  # [N, C]
            for i in range(all_centers.shape[0]):
                proto = fg_feats[fg_labels == i].mean(0)  # [C]
                fg_protos.append(proto)

            fg_protos = torch.stack(fg_protos, dim=1)  # [C, k]
            batch_fg_protos.append(fg_protos)
        fg_proto = torch.stack(batch_fg_protos, dim=0).transpose(1, 2)  # [B, k, C]

        return fg_proto

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





