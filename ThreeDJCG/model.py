from dataclasses import dataclass
from pathlib import Path

import os
import json
import pickle
import argparse
from .lib.config_joint import CONF

import numpy as np
from .models.jointnet.jointnet import JointNet

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data.scannet.model_util_scannet import ScannetDatasetConfig

# constants
DC = ScannetDatasetConfig()


class MM3DJCG(nn.Module):
    def __init__(self):
        super().__init__()

        self.set_arg()
        # setting
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(self.args.gpu)
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        # reproducibility
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.args.seed)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        VOCAB = os.path.join(CONF.PATH.DATA, "{}_vocabulary.json") # dataset_name
        dataset_name = "ScanRefer"
        vocab_path = VOCAB.format(dataset_name)
        if os.path.exists(vocab_path):
            vocabulary = json.load(open(vocab_path))
        GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, "glove.p")
        with open(GLOVE_PICKLE, "rb") as f:
            glove = pickle.load(f)
        self.model = get_model(self.args, vocabulary, glove, device)

    # def run_train(self):
    #     optimizer = optim.construct_optimizer(
    #         self.model, self.cfg
    #     )

    def set_arg(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--folder", type=str, help="Folder containing the model")
        parser.add_argument("--dataset", type=str, help="Choose a dataset: ScanRefer or ReferIt3D", default="ScanRefer")
        parser.add_argument("--gpu", type=str, help="gpu", default="0")
        # parser.add_argument("--gpu", type=str, help="gpu", default=["0"], nargs="+")
        parser.add_argument("--batch_size", type=int, help="batch size", default=8)
        parser.add_argument("--seed", type=int, default=42, help="random seed")
        
        parser.add_argument("--lang_num_max", type=int, help="lang num max", default=8)
        parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
        parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
        parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
        parser.add_argument("--num_locals", type=int, default=-1, help="Number of local objects [default: -1]")
        parser.add_argument("--num_graph_steps", type=int, default=0, help="Number of graph conv layer [default: 0]")
        
        parser.add_argument("--query_mode", type=str, default="corner", help="Mode for querying the local context, [choices: center, corner]")
        parser.add_argument("--graph_mode", type=str, default="edge_conv", help="Mode for querying the local context, [choices: graph_conv, edge_conv]")
        parser.add_argument("--graph_aggr", type=str, default="add", help="Mode for aggregating features, [choices: add, mean, max]")
        
        parser.add_argument("--min_iou", type=float, default=0.25, help="Min IoU threshold for evaluation")
        
        parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
        parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
        parser.add_argument("--no_nms", action="store_true", help="do NOT use non-maximum suppression for post-processing.")
        
        parser.add_argument("--use_tf", action="store_true", help="Enable teacher forcing")
        parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
        parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
        parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
        parser.add_argument("--use_train", action="store_true", help="Use train split in evaluation.")
        parser.add_argument("--use_last", action="store_true", help="Use the last model")
        parser.add_argument("--use_topdown", action="store_true", help="Use top-down attention for captioning.")
        parser.add_argument("--use_relation", action="store_true", help="Use object-to-object relation in graph.")
        
        parser.add_argument("--eval_caption", action="store_true", help="evaluate the reference localization results")
        parser.add_argument("--eval_detection", action="store_true", help="evaluate the object detection results")
        parser.add_argument("--eval_pretrained", action="store_true", help="evaluate the pretrained object detection results")
        
        parser.add_argument("--force", action="store_true", help="generate the results by force")
        parser.add_argument("--save_interm", action="store_true", help="Save the intermediate results")
        self.args = parser.parse_args()

    @torch.no_grad()
    def inference(self, data_dict):
        data_dict = self.model(data_dict, True)
        return data_dict

def get_model(args, vocabulary, glove, device, root=CONF.PATH.OUTPUT, eval_pretrained=False):
    # initiate model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)
    model = JointNet(
        num_class=DC.num_class,
        vocabulary=vocabulary,
        embeddings=glove,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        input_feature_dim=input_channels,
        num_proposal=args.num_proposals,
        no_caption=not args.eval_caption,
        # use_topdown=args.use_topdown,
        num_locals=args.num_locals,
        query_mode=args.query_mode,
        # num_graph_steps=args.num_graph_steps,
        # use_relation=args.use_relation,
        use_lang_classifier=False,
        no_reference=True,
        dataset_config=DC
    )

    # if eval_pretrained:
    #     # load pretrained model
    #     print("loading pretrained VoteNet...")
    #     pretrained_model = JointNet(
    #         num_class=DC.num_class,
    #         vocabulary=vocabulary,
    #         embeddings=glove,
    #         num_heading_bin=DC.num_heading_bin,
    #         num_size_cluster=DC.num_size_cluster,
    #         mean_size_arr=DC.mean_size_arr,
    #         num_proposal=args.num_proposals,
    #         input_feature_dim=input_channels,
    #         no_caption=True
    #     )

    #     pretrained_name = "PRETRAIN_VOTENET_XYZ"
    #     if args.use_color: pretrained_name += "_COLOR"
    #     if args.use_multiview: pretrained_name += "_MULTIVIEW"
    #     if args.use_normal: pretrained_name += "_NORMAL"

    #     pretrained_path = os.path.join(CONF.PATH.PRETRAINED, pretrained_name, "model.pth")
    #     pretrained_model.load_state_dict(torch.load(pretrained_path), strict=False)

    #     # mount
    #     model.backbone_net = pretrained_model.backbone_net
    #     model.vgen = pretrained_model.vgen
    #     model.proposal = pretrained_model.proposal
    # else:
    #     # load
    #     model_name = "model_last.pth" if args.use_last else "model.pth"
    #     model_path = os.path.join(root, args.folder, model_name)
    #     model.load_state_dict(torch.load(model_path), strict=False)
    #     # model.load_state_dict(torch.load(model_path))

    # multi-GPU
    if torch.cuda.device_count() > 1:
        print("using {} GPUs...".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    
    # to device
    model.to(device)

    # set mode
    model.eval()

    return model


if __name__ == "__main__":
    MM3DJCG()