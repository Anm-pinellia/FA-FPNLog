import argparse, copy, os, time, warnings, mmcv, torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmcv.parallel import MMDataParallel
from mmdet.datasets import build_dataloader, replace_ImageToTensor
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmcv.cnn import fuse_conv_bn

from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector

class DectectorPred():
    def __init__(self, args):
        self.cfg = Config.fromfile(args.config)
        self.args = args

    def predict(self):
        args = self.args
        cfg = self.cfg
        os.environ['LOCAL_RANK'] = str(0)

        # 配置文件读取
        cfg.model.pretrained = False
        # 读取neck参数
        if cfg.model.get('neck'):
            if isinstance(cfg.model.neck, list):
                for neck_cfg in cfg.model.neck:
                    if neck_cfg.get('rfp_backbone'):
                        if neck_cfg.rfp_backbone.get('pretrained'):
                            neck_cfg.rfp_backbone.pretrained = None
            elif cfg.model.neck.get('rfp_backbone'):
                if cfg.model.neck.rfp_backbone.get('pretrained'):
                    cfg.model.neck.rfp_backbone.pretrained = None

        # in case the test dataset is concatenated
        samples_per_gpu = 1
        if isinstance(cfg.data.test, dict):
            cfg.data.test.test_mode = True
            samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
            if samples_per_gpu > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                cfg.data.test.pipeline = replace_ImageToTensor(
                    cfg.data.test.pipeline)
        elif isinstance(cfg.data.test, list):
            for ds_cfg in cfg.data.test:
                ds_cfg.test_mode = True
            samples_per_gpu = max(
                [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
            if samples_per_gpu > 1:
                for ds_cfg in cfg.data.test:
                    ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

        cfg.work_dir = args.work_dir
        cfg.gpu_ids = [0]
        distributed = False

        # 创建文件以记录配置信息和评估结果
        mmcv.mkdir_or_exist(os.path.abspath(args.work_dir))


        # 构建datasets和dataloader
        dataset = build_dataset(cfg.data.val, dict(test_mode=True))           # 采用验证集作测试数据集进行评估
        data_loader = build_dataloader(dataset, samples_per_gpu=samples_per_gpu, workers_per_gpu=cfg.data.workers_per_gpu,
                                        dist=distributed, shuffle=False)

        # 清空训练中的编码器并实例化模型
        cfg.model.train_cfg = None
        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

        # 读取权重文件
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

        # 进行模型的其他设置
        # 预防模型为fp16
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        # 加快模型速度设置
        if args.fuse_conv_bn:
            model = fuse_conv_bn(model)

        model.CLASSES = checkpoint['meta']['CLASSES']
        # model.CLASSES = dataset.CLASSES

        # 开启单卡预测结果（显示，保存，显示阈值）
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir, args.show_score_thr)

        # 保存预测结果为pkl文件
        pred_results_path = os.path.join(args.work_dir, 'pred.pkl')
        print(f'\n writing results to {pred_results_path}')
        mmcv.dump(outputs, pred_results_path)


        # 对结果进行评估
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule', 'dynamic_intervals']:
                eval_kwargs.pop(key, None)
            # 添加命令行指定的评价指标
            eval_kwargs.update(dict(metric=args.eval))
            # 结果评价，（注意此处评价方法写在datasets中）
            metric = dataset.evaluate(outputs, **eval_kwargs)
            # metric = dataset.evaluate([dataset.get_ann_info(i) for i in range(len(dataset))], **eval_kwargs)    # 测试代码
            print('评估完毕，结果为:\t',metric)
            metric_dict = dict(config=args.config, metric=metric)
            # 保存评估结果
            timestamp = time.strftime('%Y年%m月%d日_%H时%M分%S秒', time.localtime())
            json_file = os.path.join(args.work_dir, f'eval_{timestamp}.json')
            mmcv.dump(metric_dict, json_file)


def parse_args():
    parser = argparse.ArgumentParser(description='训练配置')
    parser.add_argument('--config', help='配置文件目录')
    parser.add_argument('--checkpoint', help='模型权重文件')
    parser.add_argument('--work-dir', help='工作目录（权重文件和日志都将保存在这里，没有则会建立文件夹）')
    parser.add_argument('--show', default=False, help='显示结果')
    parser.add_argument('--show-score-thr', type=float, default=0.3, help='绘制的阈值')
    parser.add_argument('--show-dir', help='预测结果图片保存地址')
    parser.add_argument('--eval', type=str, nargs='+', help='指定评价指标(注意，此代码中一次只能指定一种评价metric)')
    parser.add_argument('--seed', type=int, default='723', help='指定种子确保实验的一致性')
    parser.add_argument('--fuse-conv-bn', default=False, help='开启可略微提高推测速度')
    args = parser.parse_args(['--eval', 'mAP'])
    return args


if __name__ == '__main__':
    cfg_path = r'ROI Transformer HorNetKL Seg\main_config.py'
    work_dir = r'ROI Transformer HorNetKL Seg\work_dir'
    check_point_path = r'ROI Transformer HorNetKL Seg\work_dir\epoch_12.pth'

    # 解析参数，开启训练
    args = parse_args()
    args.config = os.path.abspath(cfg_path)
    args.work_dir = os.path.abspath(work_dir)
    args.checkpoint = os.path.abspath(check_point_path)
    pred_util = DectectorPred(args)
    pred_util.predict()


