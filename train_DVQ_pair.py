import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms

import torch.backends.cudnn as cudnn

import IQADataset
import models.UIQA as UIQA
from utils import performance_fit
from utils import Fidelity_Loss, plcc_loss

import random



def parse_args():
    """Parse input arguments. """
    parser = argparse.ArgumentParser(description="UHD Image Quality Assessment")
    parser.add_argument('--gpu', help="GPU device id to use [0]", default=0, type=int)
    parser.add_argument('--n_fragment', type=int, default=12)
    parser.add_argument('--fragments_h', type=int, default=2)
    parser.add_argument('--fragments_w', type=int, default=4)
    parser.add_argument('--num_epochs',  help='Maximum number of training epochs.', default=30, type=int)
    parser.add_argument('--batch_size', help='Batch size.', default=40, type=int)
    parser.add_argument('--resize', type=int)
    parser.add_argument('--salient_patch_dimension', type=int, default=480)
    parser.add_argument('--spatial_sample_rate', type=int,default=1)
    parser.add_argument('--crop_size', type=int)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--lr_weight_L2', type=float, default=1)
    parser.add_argument('--lr_weight_pair', type=float, default=1)
    parser.add_argument('--decay_ratio', type=float, default=0.9)
    parser.add_argument('--decay_interval', type=float, default=10)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--snapshot', default='', type=str)
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--database_dir', type=str)
    parser.add_argument('--model', default='UIQA', type=str)
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--print_samples', type=int, default = 50)
    parser.add_argument('--database', default='UHD_IQA', type=str)


    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    




    torch.manual_seed(args.random_seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    gpu = args.gpu
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    decay_interval = args.decay_interval
    decay_ratio = args.decay_ratio
    snapshot = args.snapshot
    database = args.database
    print_samples = args.print_samples
    results_path = args.results_path
    database_dir = args.database_dir
    resize = args.resize
    crop_size = args.crop_size
    n_fragment = args.n_fragment
    fragments_h = args.fragments_h
    fragments_w = args.fragments_w
    salient_patch_dimension = args.salient_patch_dimension
    spatial_sample_rate = args.spatial_sample_rate

    seed = args.random_seed
    if not os.path.exists(snapshot):
        os.makedirs(snapshot)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if database == 'UHD_IQA':
        filename_list = 'csvfiles/uhd-iqa-training-metadata.csv'    
    elif database == 'NTIREVideo':
        filename_list = '/mnt/sda/fk/data/VQA/KVQ/train_data.csv'
        database_dir = '/mnt/sda/fk/data/VQA/KVQ/image_original'
    elif database == 'DVQ':
        # filename_list = '/mnt/sda/fk/data/VQA/DVQ/RQ_VQA_results_4w_videos.csv'
        # database_dir = '/mnt/sda/fk/data/VQA/DVQ/image_original'
        # filename_list = ['/mnt/sda/fk/data/VQA/DVQ/RQ_VQA_results_4w_videos.csv', '/mnt/sda/fk/data/VQA/DVQ/videos_compress_results.csv']
        # database_dir = ['/mnt/sda/fk/data/VQA/DVQ/image_original','/mnt/sda/fk/data/VQA/DVQ/RQ-inference/test_image_original/']
        filename_list = ['/data/sunwei_data/NTIRE25_5W2_Videos/RQ_VQA_results_4w_videos.csv', '/data/sunwei_data/NTIRE25_5W2_Videos/videos_compress_results.csv']
        database_dir = ['/data/sunwei_data/NTIRE25_5W2_Videos/image_4w_original','/data/sunwei_data/NTIRE25_5W2_Videos/image_compress_original']

    print(filename_list)
    
    # load the network
    if args.model == 'UIQA':
        model = UIQA.UIQA_Model(pretrained_path = args.pretrained_path)
    elif args.model == 'UVQA':
        model = UIQA.UVQA_Model(pretrained_path= args.pretrained_path)


    transforms_train = transforms.Compose([transforms.Resize(resize),
                                            transforms.RandomCrop(crop_size), 
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transforms_test = transforms.Compose([transforms.Resize(resize),
                                            transforms.CenterCrop(crop_size), 
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    




    train_dataset = IQADataset.KVQ_dataloader_pair(database_dir,
                                              filename_list,
                                              transforms_train,
                                              'DVQ',
                                              crop_size,
                                              n_fragment,
                                              seed,
                                              spatial_sample_rate)
    # train_dataset = IQADataset.UIQA_dataloader_pair(database_dir, 
    #                                                 filename_list, 
    #                                                 transforms_train, 
    #                                                 database+'_train', 
    #                                                 n_fragment, 
    #                                                 salient_patch_dimension,
    #                                                 seed)
    # test_dataset = IQADataset.UIQA_dataloader(database_dir, 
    #                                             filename_list, 
    #                                             transforms_test, 
    #                                             database+'_test',  
    #                                             n_fragment, 
    #                                             salient_patch_dimension,
    #                                             seed)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=True, 
                                                num_workers=16)

    if args.multi_gpu:
        model = torch.nn.DataParallel(model)
        model = model.to(device)
    else:
        model = model.to(device)

    
    criterion = Fidelity_Loss()
    # criterion = plcc_loss
    # criterion = nn.MSELoss().to(device)
    criterion2 = nn.MSELoss().to(device)


    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))


    optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=lr, 
                                    weight_decay=0.0000001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=decay_interval, 
                                                gamma=decay_ratio)


    print("Ready to train network")

    best_test_criterion = -1  # SROCC min
    best = np.zeros(5)

    n_train = len(train_dataset)


    for epoch in range(num_epochs):
        # train
        model.train()

        batch_losses = []
        batch_losses_each_disp = []
        session_start_time = time.time()
        for i, data_train in enumerate(train_loader):
            img_aesthetics = data_train['img_aesthetics'].to(device)
            img_distortion = data_train['img_distortion'].to(device)
            # img_saliency = data_train['img_saliency'].to(device)
            mos = data_train['y_label'][:,np.newaxis]
            mos = mos.to(device)
            # mos = data_train['y_label']
            # mos = mos.to(device).float()

            img_second_aesthetics = data_train['img_second_aesthetics'].to(device)
            img_second_distortion = data_train['img_second_distortion'].to(device)
            # img_second_saliency = data_train['img_second_saliency'].to(device)
            mos_second = data_train['y_label_second'][:,np.newaxis]
            mos_second = mos_second.to(device)
            # mos_second = data_train['y_label_second']
            # mos_second = mos_second.to(device).float()
            

            mos_output = model(img_aesthetics, img_distortion)
            mos_output_second = model(img_second_aesthetics, img_second_distortion)

            mos_output_diff = mos_output- mos_output_second
            constant =torch.sqrt(torch.Tensor([2])).to(device)
            p_output = 0.5 * (1 + torch.erf(mos_output_diff / constant))
            mos_diff = mos - mos_second
            p = 0.5 * (1 + torch.erf(mos_diff / constant))

            optimizer.zero_grad()
            loss = args.lr_weight_pair*criterion(p_output, p.detach()) + \
                args.lr_weight_L2*criterion2(mos_output, mos) + \
                args.lr_weight_L2*criterion2(mos_output_second, mos_second)
            # loss = criterion(mos_output, mos) + criterion(mos_output_second, mos_second)


            batch_losses.append(loss.item())
            batch_losses_each_disp.append(loss.item())

            loss.backward()
            optimizer.step()

            if (i+1) % print_samples == 0:
                session_end_time = time.time()
                avg_loss_epoch = sum(batch_losses_each_disp) / print_samples
                # print('Epoch: %d/%d | Step: %d/%d | Training loss: %.4f' % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, avg_loss_epoch))
                print('Epoch: {:d}/{:d} | Step: {:d}/{:d} | Training loss: {:.4f}'.format(epoch + 1, 
                                                                                            num_epochs, 
                                                                                            i + 1, 
                                                                                            len(train_dataset)//batch_size, 
                                                                                            avg_loss_epoch))
                batch_losses_each_disp = []
                print('CostTime: {:.4f}'.format(session_end_time - session_start_time))
                session_start_time = time.time()

        avg_loss = sum(batch_losses) / (len(train_dataset) // batch_size)
        # print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))
        print('Epoch {:d} averaged training loss: {:.4f}'.format(epoch + 1, avg_loss))

        scheduler.step()
        lr_current = scheduler.get_last_lr()
        print('The current learning rate is {:.06f}'.format(lr_current[0]))

        save_model_name = os.path.join(args.snapshot, 
                    args.model + '_' +  args.database + '_' + '_NR_' + str(args.spatial_sample_rate )+ '_' + str(args.crop_size) + '_' + 'epoch_%d.pth' % (epoch + 1))
        torch.save(model.module.state_dict(), save_model_name)
    
    print(database)
    print('*************************************************************************************************************************')

