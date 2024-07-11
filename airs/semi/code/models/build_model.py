import torch
import models
import os

def build_model(args):
    model = getattr(models, args.model)(args.nclasses, args.band)
    if args.GPUs:
        model.cuda()
        torch.backends.cudnn.benchmark = True
    if args.load_ckpt is not None:

        model_dict = model.state_dict()
        load_ckpt_path = os.path.join(args.root, "semi/checkpoint/" + str(args.ckpt_name), args.load_ckpt + '.pth')
        print(load_ckpt_path)
        assert os.path.isfile(load_ckpt_path), 'No checkpoint found.'
        print('Loading checkpoint......')
        checkpoint = torch.load(load_ckpt_path)
        new_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
        print('Done')

    return model
