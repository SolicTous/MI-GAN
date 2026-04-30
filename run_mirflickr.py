"""
Запуск обучения на датасете mirflickr в два этапа:
1. ComodGAN
2. MI-GAN

Использует gloo бэкенд (работает на Windows) и однопоточный режим для 1 GPU.
"""

import torch
import numpy as np
import timeit

from lib.cfg_helper import get_command_line_args, cfg_initiates
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.experiments import get_experiment
from lib.utils import exec_container


class SingleProcessExecutor(exec_container):
    """
    Версия exec_container для запуска без multiprocessing и без NCCL.
    Использует gloo бэкенд и запускается в одном процессе.
    """
    def __call__(self, RANK=0, **kwargs):
        """
        Запуск в однопоточном режиме.
        """
        self.RANK = RANK
        cfg = self.cfg
        cfguh().save_cfg(cfg)

        # Инициализация распределенного обучения с gloo бэкендом
        if cfg.env.gpu_count > 1:
            import torch.distributed as dist
            dist.init_process_group(
                backend=cfg.env.dist_backend,
                init_method=cfg.env.dist_url,
                rank=RANK,
                world_size=cfg.env.gpu_count,
            )
        else:
            # Для одной GPU инициализируем "фейковый" процесс групп с world_size=1
            import torch.distributed as dist
            try:
                dist.init_process_group(
                    backend=cfg.env.dist_backend,
                    init_method='file:///tmp/.torch_' + str(id(self)),
                    rank=0,
                    world_size=1,
                )
            except:
                # Если не получается, используем локальную инициализацию
                pass

        # Установка random seed
        if isinstance(cfg.env.rnd_seed, int):
            np.random.seed(cfg.env.rnd_seed)
            torch.manual_seed(cfg.env.rnd_seed)

        time_start = timeit.default_timer()

        para = {'RANK': RANK, 'itern_total': 0}

        for stage in self.registered_stages:
            stage_para = stage(**para)
            if stage_para is not None:
                para.update(stage_para)

        print(f'Total {timeit.default_timer() - time_start:.2f} seconds')
        
        # Очистка
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        except:
            pass
        
        self.RANK = None


def run_stage(experiment_name, phase_name):
    """
    Запуск одного этапа обучения (comodgan или migan).
    
    Args:
        experiment_name: имя эксперимента (например, 'comodgan_mirflickr512')
        phase_name: название фазы для логирования
    """
    print(f"\n{'='*60}")
    print(f"Запуск этапа: {phase_name}")
    print(f"Эксперимент: {experiment_name}")
    print(f"{'='*60}\n")
    
    # Получаем конфиг через стандартный механизм
    import sys
    sys.argv = ['main.py', '--experiment', experiment_name]
    
    cfg = get_command_line_args()
    isresume = 'resume_path' in cfg.env

    if 'train' in cfg and not isresume:
        from lib.cfg_helper import get_experiment_id
        cfg.train.experiment_id = get_experiment_id()

    cfg = cfg_initiates(cfg)
    
    # ВАЖНО: Переключаем на gloo бэкенд для Windows
    cfg.env.dist_backend = 'gloo'
    # Убеждаемся, что только 1 GPU
    if len(cfg.env.gpu_device) > 1:
        cfg.env.gpu_device = [cfg.env.gpu_device[0]]
    cfg.env.gpu_count = len(cfg.env.gpu_device)
    
    print(f"Бэкенд: {cfg.env.dist_backend}")
    print(f"GPU устройства: {cfg.env.gpu_device}")
    print(f"Количество GPU: {cfg.env.gpu_count}")

    if 'train' in cfg: 
        trainer = SingleProcessExecutor(cfg)
        tstage = get_experiment(cfg.train.exec_stage)()
        trainer.register_stage(tstage)

        # Запускаем в однопоточном режиме (RANK=0)
        trainer(0)
    else:
        evaler = SingleProcessExecutor(cfg)
        estage = get_experiment(cfg.eval.exec_stage)()
        evaler.register_stage(estage)
        if cfg.env.debug:
            evaler(0)
        else:
            evaler(0)
    
    print(f"\n{'='*60}")
    print(f"Этап {phase_name} завершен успешно!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    try:
        # Этап 1: ComodGAN
        run_stage('comodgan_mirflickr512', 'ComodGAN')
        
        # Этап 2: MI-GAN
        run_stage('migan_mirflickr512', 'MI-GAN')
        
        print("\n" + "="*60)
        print("ВСЕ ЭТАПЫ ОБУЧЕНИЯ ЗАВЕРШЕНЫ УСПЕШНО!")
        print("="*60)
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ОШИБКА при обучении: {e}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        exit(1)
