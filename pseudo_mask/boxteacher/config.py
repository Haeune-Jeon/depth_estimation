from detectron2.config import CfgNode as CN


def add_box_teacher_config(cfg):
    
    
    cfg.MODEL.BOX_TEACHER = CN()
    
    # filter and assignment
    cfg.MODEL.BOX_TEACHER.IOU_THR = 0.5
    cfg.MODEL.BOX_TEACHER.SCORE_THR = 0.0
    # EMA
    cfg.MODEL.BOX_TEACHER.MOMENTUM = 0.999
    
    # loss for pseudo masks
    # mask weight for pseudo loss
    cfg.MODEL.BOX_TEACHER.MASK_WEIGHT = 0.5
    # warmup for pseudo loss
    cfg.MODEL.BOX_TEACHER.WITH_WARMUP = True
    cfg.MODEL.BOX_TEACHER.WARMUP_ITERS = 10000
    cfg.MODEL.BOX_TEACHER.WARMUP_METHOD = "linear"
    # add avg projection loss
    cfg.MODEL.BOX_TEACHER.WITH_AVG_LOSS = False
    cfg.MODEL.BOX_TEACHER.AVG_LOSS_WEIGHT = 0.1
    # affinity loss
    cfg.MODEL.BOX_TEACHER.MASK_AFFINITY_THRESH = 0.5
    cfg.MODEL.BOX_TEACHER.MASK_AFFINITY_WEIGHT = 0.1
    # fix reduction factor for pseudo loss
    cfg.MODEL.BOX_TEACHER.FIX_REDUCTION = True

    # inference using teacher
    cfg.MODEL.BOX_TEACHER.USE_TEACHER_INFERENCE = True
    # teacher with dynamic batch norm
    cfg.MODEL.BOX_TEACHER.TEACHER_EVAL = False

    # using augmentation
    cfg.MODEL.BOX_TEACHER.USE_AUG = False
    
    # return float masks instead of binary masks
    cfg.MODEL.BOX_TEACHER.RETURN_FLOAT_MASK = False
    # mask threshold for teacher
    cfg.MODEL.BOX_TEACHER.TEACHER_MASK_THRESHOLD = 0.5
    
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
    cfg.SOLVER.OPTIMIZER = "SGD"


    # augmentation strong, weak, none
    cfg.INPUT.AUG_TYPE = "strong"
    cfg.INPUT.AUG_EXTRA = True

    # ResNet
    cfg.MODEL.RESNETS = CN()
    cfg.MODEL.RESNETS.DEPTH = 50
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.RESNETS.NORM = "FrozenBN"  # 또는 "SyncBN", "BN" 등 가능
    cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 64
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
    cfg.MODEL.RESNETS.STRIDE_IN_1X1 = True
    cfg.MODEL.RESNETS.NUM_GROUPS = 1
    cfg.MODEL.RESNETS.WIDTH_PER_GROUP = 64
    cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE = [False, False, False, False]
    cfg.MODEL.RESNETS.DEFORM_MODULATED = False
    cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS = 1
    cfg.MODEL.RESNETS.RES5_DILATION = 1
    cfg.MODEL.RESNETS.DEFORM_INTERVAL = 1
    
    # FPN 설정 (ResNet을 기반으로 한 Feature Pyramid Network)
    cfg.MODEL.FPN = CN()
    cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.FPN.OUT_CHANNELS = 256
    cfg.MODEL.FPN.FUSE_TYPE = "sum"
    cfg.MODEL.FPN.NORM = ""

    # RPN 설정 (FPN의 출력 사용)
    cfg.MODEL.RPN = CN()
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    cfg.MODEL.RPN.NMS_THRESH = 0.7
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.RPN.POSITIVE_FRACTION = 0.5
    cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
    cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = "smooth_l1"
    cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT = 1.0
    cfg.MODEL.RPN.SMOOTH_L1_BETA = 0.0
    cfg.MODEL.RPN.LOSS_WEIGHT = 1.0
    cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
    cfg.MODEL.RPN.IOU_LABELS = [0, -1, 1]
    cfg.MODEL.RPN.BOUNDARY_THRESH = -1
    cfg.MODEL.RPN.HEAD_NAME = "StandardRPNHead"
    cfg.MODEL.RPN.CONV_DIMS = [-1]  # 기본값
    cfg.MODEL.RPN.MIN_SIZE = 0

from .condinst import CondInst
from detectron2.modeling.proposal_generator import PROPOSAL_GENERATOR_REGISTRY

PROPOSAL_GENERATOR_REGISTRY.register(CondInst)