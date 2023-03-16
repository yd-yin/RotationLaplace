import matplotlib.pyplot as plt
import numpy as np
import tqdm
from Pascal3D import Pascal3D, Pascal3D_render, Pascal3D_all
from ModelNetSo3 import ModelNetSo3
from resnet import resnet50, resnet101, ResnetHead
import loss as py_loss
import torch
import os
from os.path import join, basename, dirname, abspath
import tqdm
import argparse
import dataloader_utils
import logger
import matplotlib
import json
import rotation_laplace

matplotlib.use('Agg')

torch.backends.cuda.matmul.allow_tf32 = False



dataset_dir = 'datasets'


def vmf_loss(net_out, R, overreg=1.05):
    A = net_out.view(-1, 3, 3)
    loss_v = py_loss.KL_Fisher(A, R, overreg=overreg)
    if loss_v is None:
        Rest = torch.unsqueeze(torch.eye(3, 3, device=R.device, dtype=R.dtype), 0)
        Rest = torch.repeat_interleave(Rest, R.shape[0], 0)
        return None, Rest

    Rest = py_loss.batch_torch_A_to_R(A)
    return loss_v, Rest


def get_pascal_no_warp_loaders(batch_size, train_all, voc_train, source, category=None):
    dataset = Pascal3D.Pascal3D(dataset_dir, train_all=train_all, use_warp=False, voc_train=voc_train, source=source, category=category)
    dataloader_train = torch.utils.data.DataLoader(
        dataset.get_train(False),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2 ** 32)),
        pin_memory=True,
        drop_last=True)
    dataloader_eval = torch.utils.data.DataLoader(
        dataset.get_eval(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2 ** 32)),
        pin_memory=True,
        drop_last=False)
    return dataloader_train, dataloader_eval


def get_pascal_loaders(batch_size, train_all, use_synthetic_data, use_augment, voc_train, source, category=None):
    if use_synthetic_data:
        return get_pascal_synthetic(batch_size, train_all, use_augment, voc_train, source, category)
    else:
        dataset = Pascal3D.Pascal3D(dataset_dir, train_all=train_all, use_warp=True, voc_train=voc_train, source=source, category=category)
        dataloader_train = torch.utils.data.DataLoader(
            dataset.get_train(use_augment),
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2 ** 32)),
            pin_memory=True,
            drop_last=True)
        dataloader_eval = torch.utils.data.DataLoader(
            dataset.get_eval(),
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2 ** 32)),
            pin_memory=True,
            drop_last=False)
        return dataloader_train, dataloader_eval


def get_pascal_synthetic(batch_size, train_all, use_augmentation, voc_train, source, category):
    dataset_real = Pascal3D.Pascal3D(dataset_dir, train_all=train_all, use_warp=True, voc_train=voc_train, source=source, category=category)
    train_real = dataset_real.get_train(use_augmentation)
    real_sampler = torch.utils.data.sampler.RandomSampler(train_real, replacement=False)
    dataset_rendered = Pascal3D_render.Pascal3DRendered(dataset_dir, category=category)
    rendered_size = int(0.2 * len(dataset_rendered))  # use 20% of synthetic data for training per epoch
    rendered_sampler = dataloader_utils.RandomSubsetSampler(dataset_rendered, rendered_size)
    dataset_train, sampler_train = dataloader_utils.get_concatenated_dataset([(train_real, real_sampler), (dataset_rendered, rendered_sampler)])

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=batch_size,
        num_workers=8,
        worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2 ** 32)),
        pin_memory=True,
        drop_last=True)

    dataloader_eval = torch.utils.data.DataLoader(
        dataset_real.get_eval(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2 ** 32)),
        pin_memory=True,
        drop_last=False)
    return dataloader_train, dataloader_eval


def get_modelnet_loaders(batch_size, train_all, category=None):
    dataset = ModelNetSo3.ModelNetSo3(dataset_dir, category)
    dataloader_train = torch.utils.data.DataLoader(
        dataset.get_train(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2 ** 32)),
        pin_memory=True,
        drop_last=True)
    dataloader_eval = torch.utils.data.DataLoader(
        dataset.get_eval(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2 ** 32)),
        pin_memory=True,
        drop_last=False)
    return dataloader_train, dataloader_eval


def get_optimizer(param, lr):
    opt = torch.optim.SGD(param, lr=lr)
    return opt


def load_network_weights(path, epoch, model, device):
    path = os.path.join(path, 'saved_weights', 'state_dict_{}.pkl'.format(epoch))
    with open(path, 'rb') as f:
        state_dict = torch.load(f, map_location=device)
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)



def train_model(out_dim, train_setting):
    # device = 'cpu'
    device = 'cuda'
    batch_size = train_setting.batch_size
    train_all = True  # train_all=False when decisions were made
    config = train_setting.config
    run_name = train_setting.run_name
    base = resnet101(pretrained=True, progress=True)
    if config.type == 'pascal':
        num_classes = 12 + 1  # +1 due to one indexed classes
    elif config.type == 'modelnet':
        num_classes = 10
    else:
        raise ValueError

    model = ResnetHead(base, num_classes, config.embedding_dim, 512, out_dim)

    if len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
        print('Let us use multiple GPUs:', os.environ['CUDA_VISIBLE_DEVICES'])
        model = torch.nn.DataParallel(model)
    model.to(device)

    if config.type == 'pascal':
        use_synthetic_data = config.synthetic_data
        use_augmentation = config.data_aug
        use_warp = config.warp
        voc_train = config.pascal_train
        source = config.source
        if not use_warp:
            assert (not use_synthetic_data)
            assert (not use_augmentation)
            dataloader_train, dataloader_eval = get_pascal_no_warp_loaders(batch_size, train_all, voc_train, source, train_setting.category)
        else:
            dataloader_train, dataloader_eval = get_pascal_loaders(batch_size, train_all, use_synthetic_data, use_augmentation, voc_train, source, train_setting.category)
    elif config.type == 'modelnet':
        dataloader_train, dataloader_eval = get_modelnet_loaders(batch_size, train_all, train_setting.category)
    else:
        raise Exception("Unknown config: {}".config.format())


    if isinstance(model, torch.nn.DataParallel):
        if model.module.class_embedding is None:
            finetune_parameters = model.module.head.parameters()
        else:
            finetune_parameters = list(model.module.head.parameters()) + list(model.module.class_embedding.parameters())
    else:
        if model.class_embedding is None:
            finetune_parameters = model.head.parameters()
        else:
            finetune_parameters = list(model.head.parameters()) + list(model.class_embedding.parameters())


    if config.type == 'modelnet':
        num_epochs = 50
        drop_epochs = [30, 40, 45, np.inf]
        stop_finetune_epoch = 2
    else:
        num_epochs = 120
        drop_epochs = [30, 60, 90, np.inf]
        stop_finetune_epoch = 3
    drop_idx = 0
    last_epoch = num_epochs - 1


    grids_path = join(dirname(abspath(__file__)), 'eq_grids', 'grids3.npy')
    print(f'Loading SO3 discrete grids {grids_path}')
    grids = torch.from_numpy(np.load(grids_path)).to(device)

    cur_lr = train_setting.lr
    opt = get_optimizer(finetune_parameters, lr=cur_lr)
    if config.type == 'pascal':
        class_enum = Pascal3D.PascalClasses
    else:
        class_enum = ModelNetSo3.ModelNetSo3Classes

    if train_setting.eval_only:
        log_name = 'tmp'
        num_epochs = 1
        verbose = False
        try:
            eval_epoch = os.environ['CKPT']
        except:
            eval_epoch = last_epoch
        load_network_weights(f'logs/{config.type}/{run_name}', eval_epoch, model, 'cuda')
    else:
        log_name = run_name

    log_base = 'logs'
    log_dir = '{}/{}/{}'.format(log_base, config.type, log_name)
    loggers = logger.Logger(log_dir, class_enum, config=config)

    iteration = 0
    for epoch in range(num_epochs):
        # training
        if not train_setting.eval_only:
            verbose = epoch % 20 == 0 or epoch == num_epochs - 1
            if epoch == drop_epochs[drop_idx]:
                cur_lr *= 0.1
                drop_idx += 1
                opt = get_optimizer(model.parameters(), lr=cur_lr)
            elif epoch == stop_finetune_epoch:
                opt = get_optimizer(model.parameters(), lr=cur_lr)
            logger_train = loggers.get_train_logger(epoch, verbose)
            model.train()
            for image, extrinsic, class_idx_cpu, hard, _, _ in tqdm.tqdm(dataloader_train):
                iteration += 1
                image = image.to(device)
                R = extrinsic[:, :3, :3].to(device)
                class_idx = class_idx_cpu.to(device)
                out = model(image, class_idx)
                if train_setting.loss == 'fisher':
                    losses, Rest = vmf_loss(out, R, overreg=1.025)
                elif train_setting.loss.startswith('R'):
                    losses, Rest = rotation_laplace.NLL_loss(train_setting.loss, out, R, grids)
                else:
                    raise ValueError

                if losses is not None:
                    loss = torch.mean(losses)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                else:
                    losses = torch.zeros(R.shape[0], dtype=R.dtype, device=R.device)
                logger_train.add_samples(image, losses, None, R, Rest, class_idx_cpu, hard)

                # ========== evaluation ==========
                is_drop = epoch > drop_epochs[0]
                if is_drop and train_setting.eval_freq > 0 and iteration % train_setting.eval_freq == 0:
                    logger_eval = loggers.get_validation_logger(epoch, verbose)
                    model.eval()
                    with torch.no_grad():
                        for image, extrinsic, class_idx_cpu, hard, _, _ in tqdm.tqdm(dataloader_eval):
                            image = image.to(device)
                            R = extrinsic[:, :3, :3].to(device)
                            class_idx = class_idx_cpu.to(device)
                            out = model(image, class_idx)
                            if train_setting.loss == 'fisher':
                                losses, Rest = vmf_loss(out, R)
                            elif train_setting.loss.startswith('R'):
                                losses, Rest = rotation_laplace.NLL_loss(train_setting.loss, out, R, grids)
                            else:
                                raise ValueError

                            if losses is None:
                                losses = torch.zeros(R.shape[0], dtype=R.dtype, device=R.device)
                            logger_eval.add_samples(image, losses, None, R, Rest, class_idx_cpu, hard)
                    logger_eval.finish(train_setting.eval_only, iteration=iteration)

                    model.train()       # finish evaluation
                # ===============================

            logger_train.finish()

        # evaluation
        logger_eval = loggers.get_validation_logger(epoch, verbose)
        model.eval()
        with torch.no_grad():
            for image, extrinsic, class_idx_cpu, hard, _, _ in tqdm.tqdm(dataloader_eval):
                image = image.to(device)
                R = extrinsic[:, :3, :3].to(device)
                class_idx = class_idx_cpu.to(device)
                out = model(image, class_idx)
                if train_setting.loss == 'fisher':
                    losses, Rest = vmf_loss(out, R)
                elif train_setting.loss.startswith('R'):
                    losses, Rest = rotation_laplace.NLL_loss(train_setting.loss, out, R, grids)
                else:
                    raise ValueError

                if losses is None:
                    losses = torch.zeros(R.shape[0], dtype=R.dtype, device=R.device)
                logger_eval.add_samples(image, losses, None, R, Rest, class_idx_cpu, hard)
        logger_eval.finish(train_setting.eval_only, iteration=iteration)
        if verbose:
            loggers.save_network(epoch, model)


class TrainSetting():
    def __init__(self, config, args):
        self.config = config
        print("---- Train Setting -----")
        for k, v in sorted(args.__dict__.items()):
            self.__setattr__(k, v)
            print(f"{k:20}: {v}")


class TrainConfig():
    def __init__(self, typ):
        self.type = typ

    @staticmethod
    def json_deserialize(dic):
        if dic['type'] == 'pascal':
            return PascalConfig.json_deserialize(dic)
        elif dic['type'] == 'modelnet':
            return ModelnetConfig.json_deserialize(dic)
        else:
            raise RuntimeError('Can not deserialize Train config: {}'.format(dic))

    def json_serialize(self):
        raise RuntimeError('can not serialize abstract class')


class PascalConfig(TrainConfig):
    # data_aug is bool
    # embedding_dim is int
    # synthetic_data is bool
    # warp is bool
    def __init__(self, data_aug, embedding_dim, synthetic_data, warp, pascal_train, source):
        super().__init__('pascal')
        self.data_aug = data_aug
        self.embedding_dim = embedding_dim
        self.synthetic_data = synthetic_data
        self.warp = warp
        self.pascal_train = pascal_train
        self.source = source

    @staticmethod
    def json_deserialize(dic):
        data_aug = dic['data_aug']
        embedding_dim = dic['embedding_dim']
        synthetic_data = dic['synthetic_data']
        warp = dic['warp']
        pascal_train = dic['pascal_train']
        source = dic['source']
        return PascalConfig(data_aug, embedding_dim, synthetic_data, warp, pascal_train, source)

    def json_serialize(self):
        return {'type': 'pascal',
                'data_aug': self.data_aug,
                'embedding_dim': self.embedding_dim,
                'synthetic_data': self.synthetic_data,
                'warp': self.warp,
                'pascal_train': self.pascal_train}


class ModelnetConfig(TrainConfig):
    # embedding_dim is int
    def __init__(self, embedding_dim):
        super().__init__('modelnet')
        self.embedding_dim = embedding_dim

    @staticmethod
    def json_deserialize(dic):
        return ModelnetConfig(dic['embedding_dim'])

    def json_serialize(self):
        return {'type': 'modelnet',
                'embedding_dim': self.embedding_dim}


def parse_config():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('run_name', type=str, default='dummy')
    arg_parser.add_argument('config_file', type=str)
    arg_parser.add_argument('--loss', type=str, default='RLaplace')
    arg_parser.add_argument('--net', type=str, default='resnet101')
    arg_parser.add_argument('--batch_size', type=int, default=32)
    arg_parser.add_argument('--lr', type=float, default=0.01)
    arg_parser.add_argument('--gpu', '-g', type=str, default='0')
    arg_parser.add_argument('--eval_only', '-ev', action='store_true')
    arg_parser.add_argument('--category', help='select category for ModelNet and Pascal3D+')
    arg_parser.add_argument('--eval_freq', type=int, default=1000, help='test every x iterations')
    args = arg_parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    with open(args.config_file, 'rb') as f:
        config_dict = json.load(f)
    config = TrainConfig.json_deserialize(config_dict)

    training_setting = TrainSetting(config, args)
    return training_setting



def main():
    train_setting = parse_config()
    train_model(9, train_setting)


if __name__ == '__main__':
    main()
