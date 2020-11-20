from solver.ddp_mix_solver import DDPMixSolver

# python -m torch.distributed.launch --nproc_per_node=4 main.py
if __name__ == '__main__':
    processor = DDPMixSolver(cfg_path="config/retina_free_anchor.yaml")
    processor.run()
