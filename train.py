import argparse, copy, os, time, warnings, mmcv, torch
from mmcv import Config, DictAction
from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed

from mmrotate.apis import train_detector
from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.utils import collect_env, get_root_logger

class DectectorTrain():
    def __init__(self, args):
        self.cfg = Config.fromfile(args.config)
        self.args = args

    def train(self):
        args = self.args
        cfg = self.cfg
        os.environ['LOCAL_RANK'] = str(0)
        # 配置文件读取

        cfg.work_dir = args.work_dir
        cfg.resume_from = args.resume_from
        cfg.auto_resume = args.auto_resume
        cfg.gpu_ids = [0]
        distributed = False
        if 'runner' in cfg:
            if cfg.runner.get('max_epochs'):
                cfg.runner.max_epochs = args.epochs
        cfg.total_epochs = args.epochs

        # create work_dir
        mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

        # 保存config配置信息
        cfg.dump(os.path.join(cfg.work_dir, os.path.basename(args.config)))

        # 初始化logger用于显示和保存训练信息
        timestamp = time.strftime('%Y年%m月%d日_%H时%M分%S秒', time.localtime())
        log_file = os.path.join(cfg.work_dir, f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

        # init the meta dict to record some important information such as
        # environment info and seed, which will be logged
        # 保存相关信息

        # 记录环境配置信息
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '=' * 60 + '\n'
        logger.info('本次训练环境配置信息:\n' + dash_line + env_info + '\n' + dash_line)
        logger.info(f'分布式训练: {distributed}')
        logger.info(f'训练配置信息:\n{cfg.pretty_text}')

        # 设置随机种子点，保证随机结果一致
        seed = init_random_seed(args.seed)
        set_random_seed(seed, deterministic=True)
        cfg.seed = seed
        logger.info(f'种子点设置为 {seed}')

        # 保存字典信息
        meta = dict()
        meta['env_info'] = env_info
        meta['config'] = cfg.pretty_text
        meta['seed'] = seed
        meta['exp_name'] = os.path.basename(args.config)

        # 模型初始化
        model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        model.train()
        model.init_weights()

        # 训练datasets构建
        datasets = [build_dataset(cfg.data.train)]
        # 若加入验证流程，则在datasets中加入对应datasets
        if len(cfg.workflow) == 2:
            val_dataset = copy.deepcopy(cfg.data.val)
            val_dataset.pipeline = cfg.data.train.pipeline
            datasets.append(build_dataset(val_dataset,  dict(test_mode=True)))

        # 权重文件配置信息中，加入meta关键字，保存训练的mmdet版本和训练数据的类别名称
        if cfg.checkpoint_config is not None:
            cfg.checkpoint_config.meta = dict(mmdet_version=__version__, CLASSES=datasets[0].CLASSES)

        # 设置模型CLASSES变量
        model.CLASSES = datasets[0].CLASSES

        # 开始训练
        # validate是否在训练过程中评价
        train_detector(model, datasets, cfg, distributed=distributed, validate=True, timestamp=timestamp, meta=meta)

def parse_args():
    parser = argparse.ArgumentParser(description='训练配置')
    parser.add_argument('--config', help='配置文件目录')
    parser.add_argument('--work-  +dir', help='工作目录（权重文件和日志都将保存在这里，没有则会建立文件夹）')
    parser.add_argument('--epochs', default=12, help='训练轮数')
    parser.add_argument('--resume-from', help='继续训练的权重文件')
    parser.add_argument('--auto-resume', default=False, help='是否自动从最后一次训练结果继续')
    parser.add_argument('--seed', type=int, default='723', help='random seed')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    cfg_path = r'ReDet\main_config.py'
    work_dir = r'ReDet\work_dir'

    # 解析参数，开启训练
    args = parse_args()
    args.config = os.path.abspath(cfg_path)
    args.work_dir = os.path.abspath(work_dir)
    train_util = DectectorTrain(args)
    train_util.train()


