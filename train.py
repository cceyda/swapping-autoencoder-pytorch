import data
import models
import optimizers
from options import TrainOptions
from util import IterationCounter
from util import Visualizer
from util import MetricTracker
from evaluation import GroupEvaluator
from accelerate import Accelerator
import accelerate

import torch
import time, gc

# Timing utilities
start_time = None

def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()

def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))
    
    

# ddp_kwargs=accelerate.DistributedDataParallelKwargs( find_unused_parameters = True)
accelerator = Accelerator(fp16 =True,device_placement=False)
print(accelerator.distributed_type)
opt = TrainOptions().parse()
dataset = data.create_dataset(opt)
opt.dataset = dataset
iter_counter = IterationCounter(opt)
visualizer = Visualizer(opt)
metric_tracker = MetricTracker(opt)
evaluators = GroupEvaluator(opt)

model = models.create_model(opt,accelerator)

optimizer = optimizers.create_optimizer(opt, model)
optimizer.accelerator=accelerator
print('pre acc')
print(optimizer.optimizer_G)
print(optimizer.optimizer_D)

optimizer.optimizer_G=accelerator.prepare(optimizer.optimizer_G)
optimizer.optimizer_D=accelerator.prepare(optimizer.optimizer_D)
print('post acc')
print(optimizer.optimizer_G)
print(optimizer.optimizer_D)

dataset.dataloader = accelerator.prepare(dataset.dataloader)

start_timer()
while not iter_counter.completed_training():
    with iter_counter.time_measurement("data"):
        cur_data = next(dataset)

    with iter_counter.time_measurement("train"):
        losses = optimizer.train_one_step(cur_data, iter_counter.steps_so_far)
        metric_tracker.update_metrics(losses, smoothe=True)

    with iter_counter.time_measurement("maintenance"):
        if iter_counter.needs_printing():
            visualizer.print_current_losses(iter_counter.steps_so_far,
                                            iter_counter.time_measurements,
                                            metric_tracker.current_metrics())
            if iter_counter.steps_so_far!=0:
                end_timer_and_print("Default precision:")

        if iter_counter.needs_displaying():
            visuals = optimizer.get_visuals_for_snapshot(cur_data)
            visualizer.display_current_results(visuals,
                                               iter_counter.steps_so_far)

        if iter_counter.needs_evaluation():
            metrics = evaluators.evaluate(
                model, dataset, iter_counter.steps_so_far)
            metric_tracker.update_metrics(metrics, smoothe=False)

        if iter_counter.needs_saving():
            optimizer.save(iter_counter.steps_so_far)

        if iter_counter.completed_training():
            break

        iter_counter.record_one_iteration()

print('Training finished.')
