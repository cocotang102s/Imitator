import os  # 导入操作系统相关功能
import argparse, os, sys, datetime, glob, importlib  # 导入命令行解析、系统路径、日期时间、文件通配符、动态导入等模块
from omegaconf import OmegaConf  # Ω配置文件读取与合并
import torch  # PyTorch 主库
from torch.utils.data import DataLoader, Dataset  # 数据加载与自定义数据集
from pytorch_lightning import seed_everything  # Lightning 的随机种子设定
from pytorch_lightning.trainer import Trainer  # Lightning 的训练器
import pytorch_lightning as pl  # Lightning 简写
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor, EarlyStopping  # Lightning 回调
from pytorch_lightning.utilities.distributed import rank_zero_only  # 仅主进程执行
from imitator.utils.custom_callbacks import Callback, Save_results  # 自定义回调（与上一行名字相同，会覆盖）

torch.backends.cudnn.deterministic = True  # 固定 CUDNN 结果，保证可复现
torch.backends.cudnn.benchmark = False  # 关闭 CUDNN 基准模式，避免随机性

def get_obj_from_str(string, reload=False):  # 根据字符串获取对象
    module, cls = string.rsplit(".", 1)  # 拆分模块名和类名
    if reload:  # 如果需要重新加载
        module_imp = importlib.import_module(module)  # 导入模块
        importlib.reload(module_imp)  # 重新加载模块
    return getattr(importlib.import_module(module, package=None), cls)  # 返回对象引用


def get_parser(**parser_kwargs):  # 构建命令行参数解析器
    def str2bool(v):  # 将字符串转换为布尔值
        if isinstance(v, bool):  # 若已是布尔
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):  # 真值集合
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):  # 假值集合
            return False
        else:  # 其它非法输入
            raise argparse.ArgumentTypeError("Boolean value expected.")  # 抛出错误

    parser = argparse.ArgumentParser(**parser_kwargs)  # 创建解析器
    parser.add_argument(  # 日志目录后缀
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(  # 从某目录或检查点恢复
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(  # 基础配置文件路径
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(  # 是否训练
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(  # 是否跳过测试
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument("-p", "--project", help="name of new or path to existing project")  # 项目名称
    parser.add_argument(  # 调试模式
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(  # 随机种子
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(  # 文件名后再加后缀
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )

    return parser  # 返回解析器实例


def nondefault_trainer_args(opt):  # 找出 Trainer 非默认参数
    parser = argparse.ArgumentParser()  # 临时解析器
    parser = Trainer.add_argparse_args(parser)  # 加入 Trainer 参数
    args = parser.parse_args([])  # 获得默认参数
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))  # 返回与默认不同的参数名列表


def instantiate_from_config(config):  # 根据配置实例化对象
    if not "target" in config:  # 必须存在 target 键
        raise KeyError("Expected key `target` to instantiate.")  # 否则报错
    return get_obj_from_str(config["target"])(**config.get("params", dict()))  # 调用构造函数并返回实例


class WrappedDataset(Dataset):  # 将任意对象包装成 PyTorch Dataset
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):  # 保存原始数据
        self.data = dataset

    def __len__(self):  # 返回数据长度
        return len(self.data)

    def __getitem__(self, idx):  # 获取数据项
        return self.data[idx]


class DataModuleFromConfig(pl.LightningDataModule):  # 根据配置初始化 LightningDataModule
    def __init__(self, batch_size, train=None, validation=None, test=None, test_unseen=None,
                 wrap=False, num_workers=None):  # 初始化
        super().__init__()
        self.batch_size = batch_size  # 批大小
        self.dataset_configs = dict()  # 存储数据集配置
        self.num_workers = num_workers if num_workers is not None else batch_size*2  # 工作进程数
        if train is not None:  # 训练集配置
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader  # 绑定 dataloader 方法
        if validation is not None:  # 验证集配置
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:  # 测试集配置
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        if test_unseen is not None:  # 未见测试集配置
            self.dataset_configs["test_unseen"] = test_unseen
            self._test_unseen_dataloader = self._test_unseen_dataloader
        self.wrap = wrap  # 是否包装为 Dataset

    def prepare_data(self):  # 下载/预处理数据（分布式仅主进程执行）
        for data_cfg in self.dataset_configs.values():  # 遍历所有配置
            instantiate_from_config(data_cfg)  # 实例化一次触发准备逻辑

    def setup(self, stage=None):  # 构建真正的数据集实例
        self.datasets = dict(  # 生成每个阶段数据集
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:  # 如果需要包装
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):  # 训练 dataloader
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def _val_dataloader(self):  # 验证 dataloader
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def _test_dataloader(self):  # 测试 dataloader
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers)
    
    def _test_unseen_dataloader(self):  # 测试未见 dataloader
        return DataLoader(self.datasets["test_unseen"], batch_size=self.batch_size,
                          num_workers=self.num_workers)

class SetupCallback(Callback):  # 训练前回调，用于创建目录与保存配置
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume  # 是否恢复训练
        self.now = now  # 当前时间字符串
        self.logdir = logdir  # 日志目录
        self.ckptdir = ckptdir  # 检查点目录
        self.cfgdir = cfgdir  # 配置文件目录
        self.config = config  # 项目配置
        self.lightning_config = lightning_config  # Lightning 配置

    def on_pretrain_routine_start(self, trainer, pl_module):  # 训练前触发
        if trainer.global_rank == 0:  # 仅主进程执行
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)  # 创建日志目录
            os.makedirs(self.ckptdir, exist_ok=True)  # 创建检查点目录
            os.makedirs(self.cfgdir, exist_ok=True)  # 创建配置目录

            print("Project config")  # 打印项目配置
            print(self.config.pretty())
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))  # 保存项目配置

            print("Lightning config")
            print(self.lightning_config.pretty())
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))  # 保存 Lightning 配置

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):  # 如果不是恢复且目录已存在
                dst, name = os.path.split(self.logdir)  # 目标目录
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)  # 移动目录避免冲突
                except FileNotFoundError:
                    pass  # 如果目录消失则忽略

if __name__ == "__main__":  # 脚本入口
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")  # 当前时间戳
    sys.path.append(os.getcwd())  # 将当前工作目录加入模块搜索路径

    parser = get_parser()  # 获取参数解析器
    parser = Trainer.add_argparse_args(parser)  # 加入 Trainer 参数

    opt, unknown = parser.parse_known_args()  # 解析命令行
    seed_everything(opt.seed)  # 全局随机种子

    if opt.name and opt.resume:  # 同时指定 -n 和 -r 不合法
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    first_run=True  # 是否首次运行
    if opt.resume:  # 如果指定恢复
        print("\nResume training from the model", opt.resume, "\n")
        first_run=False
        if not os.path.exists(opt.resume):  # 路径不存在
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):  # 指定具体 ckpt 文件
            paths = opt.resume.split("/")
            idx = len(paths)-paths[::-1].index("logs")+1
            logdir = "/".join(paths[:idx])  # 日志目录
            ckpt = opt.resume  # 直接使用该 ckpt
        else:
            assert os.path.isdir(opt.resume), opt.resume  # 必须是目录
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")  # 默认 last.ckpt

        opt.resume_from_checkpoint = ckpt  # 写回参数
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))  # 找到旧配置
        opt.base = base_configs+opt.base  # 把旧配置放前面
        _tmp = logdir.split("/")
        nowname = _tmp[_tmp.index("logs")+1]  # 使用旧的 run 名称
    else:  # 新运行
        if opt.name:  # 如果给定自定义名称
            name = "_"+opt.name
        elif opt.base:  # 否则用配置文件名
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_"+cfg_name
        else:
            name = ""
        nowname = now+name+opt.postfix  # 最终 run 名称
        logdir = os.path.join(os.getenv("LOGHOME"), "logs", nowname)  # 日志根目录


    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]  # 读取所有基础配置
        cli = OmegaConf.from_dotlist(unknown)  # 命令行 dot list 覆盖
        config = OmegaConf.merge(*configs, cli)  # 合并配置
        lightning_config = config.pop("lightning", OmegaConf.create())  # 分离 Lightning 部分
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())  # 取 trainer 配置
        # default to ddp
        # trainer_config["distributed_backend"] = "ddp"
        for k in nondefault_trainer_args(opt):  # 命令行覆盖 Trainer 配置
            trainer_config[k] = getattr(opt, k)
        if not "gpus" in trainer_config:  # 如果未设置 GPU
            del trainer_config["distributed_backend"]  # 删分布式后端
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]  # GPU 信息
            print(f"Running on GPUs {gpuinfo}")
            cpu = False

        trainer_opt = argparse.Namespace(**trainer_config)  # 转为命名空间
        lightning_config.trainer = trainer_config  # 写回配置

        # added by bala
        if first_run:  # 首次运行，修改实验命名
            model_config = config.get("model", OmegaConf.create())  # 模型配置
            model_opt = argparse.Namespace(**model_config)  # 命名空间
            experiment_name = "_ADAM_nep%s" % (trainer_opt.max_epochs)  # 自定义后缀
            nowname = now+name+experiment_name+opt.postfix  # 更新 run 名称
            logdir = os.path.join(os.getenv("LOGHOME"), "logs", nowname)  # 更新目录

        # code moved here by bala
        ckptdir = os.path.join(logdir, "checkpoints")  # 检查点目录
        cfgdir = os.path.join(logdir, "configs")  # 配置目录


        # data
        data = instantiate_from_config(config.data)  # 根据配置实例化 DataModule
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        # data.prepare_data()
        data.setup()  # 调用 setup，准备数据

        # model
        model = instantiate_from_config(config.model)  # 根据配置加载模型

        # trainer and callbacks
        trainer_kwargs = dict()  # 训练器额外参数

        # default logger configs
        # NOTE wandb < 0.10.0 interferes with shutdown
        # wandb >= 0.10.0 seems to fix it but still interferes with pudb
        # debugging (wrongly sized pudb ui)
        # thus prefer testtube for now
        default_logger_cfgs = {  # 默认日志器配置集合
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
            "tensorboard": {
                "target": "pytorch_lightning.loggers.TensorBoardLogger",
                "params": {
                    "name": "tb",
                    "save_dir": logdir,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["tensorboard"]  # 默认用 TensorBoard
        logger_cfg = lightning_config.logger or OmegaConf.create()  # 如果有自定义 logger
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)  # 合并配置
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)  # 实例化 logger

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {  # 默认 ModelCheckpoint 配置
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }

        if hasattr(model, "monitor"):  # 如果模型指定监控指标
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            # commit modififed the code save the top 1, since every pixelcnn model is 4Gb
            save_top_k = trainer_config.get("save_top_k", 1)
            default_modelckpt_cfg["params"]["save_top_k"] = save_top_k  # 只保留最优 1 个 ckpt

        modelckpt_cfg = lightning_config.modelcheckpoint or OmegaConf.create()  # 合并外部配置
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)  # 加入回调

        # add callback which sets up log directory
        default_callbacks_cfg = {  # 默认回调集合
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    #"log_momentum": True
                }
            },
        }

        early_stopping_params = trainer_config.get("early_stopping_metric")  # 若配置早停
        if early_stopping_params is not None:
            early_stopping_params["monitor"] = model.monitor  # 监控指标
            early_stopping_cfg = {
                "target": "pytorch_lightning.callbacks.EarlyStopping",
                "params": early_stopping_params
            }
            default_callbacks_cfg["early_stopping"] = early_stopping_cfg
            print("early_stopping_params", early_stopping_params)

        callbacks_cfg = lightning_config.callbacks or OmegaConf.create()  # 外部回调配置
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)  # 合并
        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]  # 实例化

        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)  # 创建 Trainer

        # trainer = Trainer.from_argparse_args(trainer_opt)
        # configure learning rate
        # bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        bs, base_lr = config.data.params.batch_size, config.model.params.lr  # 批大小与基础学习率
        if not cpu:
            print(lightning_config.trainer.gpus)
            ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))  # GPU 数目
        else:
            ngpu = 1
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches or 1  # 梯度累积
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches  # 写回
        # model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        model.learning_rate = base_lr  # 设置最终学习率
        # print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
        #     model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))

        # allow checkpointing via USR1
        def melk(*args, **kwargs):  # 自定义信号处理函数，手动存 ckpt
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):  # 自定义信号处理函数，进入调试
            if trainer.global_rank == 0:
                import pudb; pudb.set_trace()

        import signal  # 信号模块
        signal.signal(signal.SIGUSR1, melk)  # 绑定 USR1
        signal.signal(signal.SIGUSR2, divein)  # 绑定 USR2

        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 可训练参数量统计
        print()
        print("Printed number of trainable params", pytorch_total_params)
        print()
        model.summarize()  # Lightning 自带模型摘要
        print()
        model.summarize(mode='full')
        print()

        # run
        if opt.train:  # 如果开启训练
            try:  
                trainer.fit(model, data)  # 执行训练
            except Exception:
                melk()  # 出错时强制存 ckpt
                raise "training failed"  # 抛出异常

    except Exception:  # 捕获任何异常
        if opt.debug and trainer.global_rank==0:  # 若调试模式，进入调试器
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise  # 重新抛出异常
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank==0:  # 调试模式下归档
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)  # 移动日志目录
