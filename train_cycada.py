import time
import datetime
import torch
import torch.distributed as dist

from cycada.options.cycada_options import CycadaOptions
from cycada.data import create_dataset
from cycada.models import create_model
from cycada.util.visualizer import Visualizer

from bms.utils import timeSince, get_score, print_rank_0


def evaluate(opt, dataset, model):
    model.eval()
    img_types = ['real_A', 'real_B', 'fake_A', 'fake_B']
    predictions = {img_type:{} for img_type in img_types}
    start = end = time.time()
    for i, data in enumerate(dataset):
        model.set_input(data)
        preds = model.predict()
        for img_type in img_types:
            predictions[img_type].update(preds[img_type])
        if i % opt.print_freq == 0 or i == (len(dataset.dataloader)-1):
            print_rank_0('EVAL: [{0}/{1}]  Elapsed {remain:s} '
                  .format(i, len(dataset.dataloader),
                          remain=timeSince(start, float(i+1)/len(dataset.dataloader))))
    gathered_preds = [None for i in range(dist.get_world_size())]
    dist.all_gather_object(gathered_preds, predictions)
    if opt.local_rank != 0:
        return
    # Gather the predictions
    predictions = {img_type:{} for img_type in img_types}
    for img_type in img_types:
        for preds in gathered_preds:
            predictions[img_type].update(preds[img_type])
    # Evaluate
    formats = ['atomtok']
    for img_type in img_types:
        scores = {}
        preds = predictions[img_type]
        if img_type in ['real_A', 'fake_B']:
            df = dataset.dataset.df_A
        else:
            df = dataset.dataset.df_B
        for format_ in formats:
            text_preds = [preds[i][format_]['best'] for i in range(len(preds))]
            scores['smiles'], scores['smiles_em'] = get_score(df['SMILES'].values, text_preds)
        print(img_type, scores)
    return


if __name__ == '__main__':
    opt = CycadaOptions().parse()   # get training options
    opt.cycada = True
    
    if opt.local_rank != -1:
        dist.init_process_group(backend='gloo', init_method='env://', timeout=datetime.timedelta(0, 7200))
        torch.cuda.set_device(opt.local_rank)
        torch.backends.cudnn.benchmark = True
    
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print_rank_0('The number of training images = %d' % dataset_size)
    
    valid_dataset = create_dataset(opt, phase='valid')
    print_rank_0('The number of valid images = %d' % len(valid_dataset))

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    
    if opt.local_rank == 0:
        visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    
#     evaluate(opt, valid_dataset, model)
    model.train()

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        
        if opt.local_rank != -1:
            dist.barrier()
            dataset.sampler.set_epoch(epoch)
        
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        if opt.local_rank == 0:         # reset the visualizer: make sure it saves the results to HTML at least once every epoch
            visualizer.reset()
        
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            
            if (i+1) % opt.print_freq == 0:
                evaluate(opt, valid_dataset, model)
                model.train()
            
            if opt.local_rank == 0:
                if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                    save_result = total_iters % opt.update_html_freq == 0
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)

            iter_data_time = time.time()
        
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
