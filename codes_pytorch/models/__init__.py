from .model import *
import logging


def build_model(opt):
    if opt['is_init']:
        if not opt.get("is_single", False):
            m = InitModel(opt)
        else:
            m = InitSingle(opt)
    else:
        m = FinetuneModel(opt)
    logger = logging.getLogger('base')
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m

