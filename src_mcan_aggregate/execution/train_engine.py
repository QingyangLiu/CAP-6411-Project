# ------------------------------------------------------------------
# NuScenes-QA
# Written by Tianwen Qian https://github.com/qiantianwen/NuScenes-QA
# ------------------------------------------------------------------

import os, torch, datetime, time
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from src.models.model_loader import ModelLoader
from src.utils.optim import get_optim, adjust_lr
from src.execution.test_engine import test_engine, ckpt_proc

import ipdb


def train_engine(__C, dataset, dataset_eval=None):

    data_size = dataset.data_size
    token_size = dataset.token_size
    ans_size = dataset.ans_size
    pretrained_emb = dataset.pretrained_emb
    #print(data_size)
    #print(token_size)
    #print(ans_size)
    #print(pretrained_emb.shape)
    #input('continue')



    pre_net=ModelLoader(__C).Net(__C, pretrained_emb, token_size, ans_size)
    ckpt=torch.load('/home/qi940700/Desktop/NuScenes-QA-new-v2/epoch13.pkl')
    if __C.N_GPU > 1:
        pre_net=nn.DataParallel(pre_net, device_ids=__C.DEVICES)
        pre_net.load_state_dict(ckpt_proc(ckpt['state_dict']))
    else:
        pre_net.load_state_dict(ckpt['state_dict'])
    
    #pre_net.requires_grad=False
    #for param in pre_net.parameters():
     #   param.requires_grad=False

    
    
    __C.USE_BBOX_FEAT=False
    __C.FEAT_SIZE['OBJ_FEAT_SIZE']=(100, 1024)
    net = ModelLoader(__C).Net(
        __C,
        pretrained_emb,
        token_size,
        ans_size
    )

    pre_net.cuda()
    pre_net.train()

    net.cuda()
    net.train()

    if __C.N_GPU > 1:
        net = nn.DataParallel(net, device_ids=__C.DEVICES)

    # Define Loss Function
    loss_fn = eval('torch.nn.' + __C.LOSS_FUNC_NAME_DICT[__C.LOSS_FUNC] + "(reduction='" + __C.LOSS_REDUCTION + "').cuda()")

    # Load checkpoint if resume training
    if __C.RESUME:
        print(' ========== Resume training')

        if __C.CKPT_PATH is not None:
            print('Warning: Now using CKPT_PATH args, '
                  'CKPT_VERSION and CKPT_EPOCH will not work')

            path = __C.CKPT_PATH
        else:
            path = __C.CKPTS_PATH + \
                   '/ckpt_' + __C.CKPT_VERSION + \
                   '/epoch' + str(__C.CKPT_EPOCH) + '.pkl'

        # Load the network parameters
        print('Loading ckpt from {}'.format(path))
        ckpt = torch.load(path)
        print('Finish!')

        if __C.N_GPU > 1:
            net.load_state_dict(ckpt_proc(ckpt['state_dict']))
        else:
            net.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt['epoch']

        # Load the optimizer paramters
        optim = get_optim(__C, net, data_size, ckpt['lr_base'], model2=pre_net)
        optim._step = int(data_size / __C.BATCH_SIZE * start_epoch)
        optim.optimizer.load_state_dict(ckpt['optimizer'])
        
        if ('ckpt_' + __C.VERSION) not in os.listdir(__C.CKPTS_PATH):
            os.mkdir(__C.CKPTS_PATH + '/ckpt_' + __C.VERSION)

    else:
        if ('ckpt_' + __C.VERSION) not in os.listdir(__C.CKPTS_PATH):
            #shutil.rmtree(__C.CKPTS_PATH + '/ckpt_' + __C.VERSION)
            os.mkdir(__C.CKPTS_PATH + '/ckpt_' + __C.VERSION)

        optim = get_optim(__C, net, data_size, model2=pre_net)
        start_epoch = 0

    loss_sum = 0
    named_params = list(net.named_parameters())
    grad_norm = np.zeros(len(named_params))

    dataloader = Data.DataLoader(
        dataset,
        batch_size=__C.BATCH_SIZE,
        shuffle=True,
        num_workers=__C.NUM_WORKERS,
        pin_memory=__C.PIN_MEM,
        drop_last=True
    )

    logfile = open(
        __C.LOG_PATH +
        '/log_run_' + __C.VERSION + '.txt',
        'a+'
    )
    logfile.write(str(__C))
    logfile.close()
    #=test_engine(__C, dataset_eval, state_dict=net.state_dict(), save_eval_result = False,pre_net=pre_net)
    # Training script
    print(start_epoch)
    print(__C.MAX_EPOCH)
    for epoch in range(start_epoch, __C.MAX_EPOCH):
        pre_net.train()
        net.train()

        # Save log to file
        logfile = open(
            __C.LOG_PATH +
            '/log_run_' + __C.VERSION + '.txt',
            'a+'
        )
        logfile.write(
            '=====================================\nnowTime: ' +
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
            '\n'
        )
        logfile.close()

        # Learning Rate Decay
        if epoch in __C.LR_DECAY_LIST:
            adjust_lr(optim, __C.LR_DECAY_R)

        time_start = time.time()
        # Iteration
        for step, (
                obj_feat_iter,
                bbox_feat_iter,
                ques_ix_iter,
                ans_iter
        ) in enumerate(dataloader):
            optim.zero_grad()
            
            obj_feat_iter = obj_feat_iter.cuda()
            bbox_feat_iter = bbox_feat_iter.cuda()
            ques_ix_iter = ques_ix_iter.cuda()
            ans_iter = ans_iter.cuda()

            loss_tmp = 0
            for accu_step in range(__C.GRAD_ACCU_STEPS):
                loss_tmp = 0

                sub_obj_feat_iter = \
                    obj_feat_iter[accu_step * __C.SUB_BATCH_SIZE:
                                (accu_step + 1) * __C.SUB_BATCH_SIZE]
                sub_bbox_feat_iter = \
                    bbox_feat_iter[accu_step * __C.SUB_BATCH_SIZE:
                                (accu_step + 1) * __C.SUB_BATCH_SIZE]
                sub_ques_ix_iter = \
                    ques_ix_iter[accu_step * __C.SUB_BATCH_SIZE:
                                (accu_step + 1) * __C.SUB_BATCH_SIZE]
                sub_ans_iter = \
                    ans_iter[accu_step * __C.SUB_BATCH_SIZE:
                            (accu_step + 1) * __C.SUB_BATCH_SIZE]


                obj_B, obj_N, obj_C=sub_obj_feat_iter.shape  
                bbox_B, bbox_N, bbox_C=sub_bbox_feat_iter.shape  

                sub_obj_feat_iter=torch.reshape(sub_obj_feat_iter, (int(obj_B*obj_N/100), 100, obj_C))
                sub_bbox_feat_iter=torch.reshape(sub_bbox_feat_iter, (int(bbox_B*bbox_N/100), 100, bbox_C))
                sub_ques_ix_iter_temp=sub_ques_ix_iter.repeat_interleave(10, dim=0)
                #print(sub_obj_feat_iter.shape)
                #print(sub_ques_ix_iter.shape)
                #input('continue')
                #print(sub_bbox_feat_iter.shape)
                #input('continue')
                __C.USE_BBOX_FEAT=True
                #__C.HIDDEN_SIZE=512
                aggregate=pre_net(sub_obj_feat_iter, sub_bbox_feat_iter, sub_ques_ix_iter_temp, return_norm_only=True)
                
                aggregate=torch.reshape(aggregate, (obj_B, 10, -1))

                #print(aggregate.shape)
                #print(sub_ques_ix_iter.shape)
                #input('continue')
                __C.USE_BBOX_FEAT=False
                #__C.HIDDEN_SIZE=1024
                pred = net(
                    aggregate,
                    sub_bbox_feat_iter,
                    sub_ques_ix_iter
                )

                loss_item = [pred, sub_ans_iter]
                loss_nonlinear_list = __C.LOSS_FUNC_NONLINEAR[__C.LOSS_FUNC]
                for item_ix, loss_nonlinear in enumerate(loss_nonlinear_list):
                    if loss_nonlinear in ['flat']:
                        loss_item[item_ix] = loss_item[item_ix].view(-1)
                    elif loss_nonlinear:
                        loss_item[item_ix] = eval('F.' + loss_nonlinear + '(loss_item[item_ix], dim=1)')

                loss = loss_fn(loss_item[0], loss_item[1])
                if __C.LOSS_REDUCTION == 'mean':
                    # only mean-reduction needs be divided by grad_accu_steps
                    loss /= __C.GRAD_ACCU_STEPS
                loss.backward()

                loss_tmp += loss.cpu().data.numpy() * __C.GRAD_ACCU_STEPS
                loss_sum += loss.cpu().data.numpy() * __C.GRAD_ACCU_STEPS

            if dataset_eval is not None:
                mode_str = __C.SPLIT['train'] + '->' + __C.SPLIT['val']
            else:
                mode_str = __C.SPLIT['train'] + '->' + __C.SPLIT['test']

            print("\r[Version %s][Dataset %s][Epoch %2d][Step %4d/%4d][%s] Loss: %.4f, Lr: %.2e" % (
                __C.VERSION,
                __C.MODEL_USE,
                epoch + 1,
                step,
                int(data_size / __C.BATCH_SIZE),
                mode_str,
                loss_tmp / __C.SUB_BATCH_SIZE,
                optim._rate
            ), end='          ')

            # Gradient norm clipping
            if __C.GRAD_NORM_CLIP > 0:
                nn.utils.clip_grad_norm_(
                    net.parameters(),
                    __C.GRAD_NORM_CLIP
                )

            # Save the gradient information
            for name in range(len(named_params)):
                norm_v = torch.norm(named_params[name][1].grad).cpu().data.numpy() \
                    if named_params[name][1].grad is not None else 0
                grad_norm[name] += norm_v * __C.GRAD_ACCU_STEPS


                # print('Param %-3s Name %-80s Grad_Norm %-20s'%
                #       (str(grad_wt),
                #        params[grad_wt][0],
                #        str(norm_v)))

            optim.step()

        time_end = time.time()
        elapse_time = time_end-time_start
        print('Finished in {}s'.format(int(elapse_time)))
        epoch_finish = epoch + 1

        # Save checkpoint
        if __C.N_GPU > 1:
            state = {
                'state_dict': net.module.state_dict(),
                'optimizer': optim.optimizer.state_dict(),
                'lr_base': optim.lr_base,
                'epoch': epoch_finish
            }
        else:
            state = {
                'state_dict': net.state_dict(),
                'optimizer': optim.optimizer.state_dict(),
                'lr_base': optim.lr_base,
                'epoch': epoch_finish
            }
            state_2={
                'state_dict': pre_net.state_dict(),
                'optimizer': optim.optimizer.state_dict(),
                'lr_base': optim.lr_base,
                'epoch': epoch_finish
            }
            print(__C.VERSION)
        torch.save(
            state,
            __C.CKPTS_PATH +
            '/ckpt_' + __C.VERSION +
            '/epoch' + str(epoch_finish) +
            '.pkl'
        )

        torch.save(state_2, __C.CKPTS_PATH +
            '/ckpt_' + __C.VERSION +
            '/epoch' + str(epoch_finish) +
            '_prenet.pkl')

        # Logging
        logfile = open(
            __C.LOG_PATH +
            '/log_run_' + __C.VERSION + '.txt',
            'a+'
        )
        logfile.write(
            'Epoch: ' + str(epoch_finish) +
            ', Loss: ' + str(loss_sum / data_size) +
            ', Lr: ' + str(optim._rate) + '\n' +
            'Elapsed time: ' + str(int(elapse_time)) + 
            ', Speed(s/batch): ' + str(elapse_time / step) +
            '\n\n'
        )
        logfile.close()

        # Eval after every epoch
        if epoch % __C.EVAL_FREQUENCY == 0:
        # if dataset_eval is not None:
            test_engine(
                __C,
                dataset_eval,
                state_dict=net.state_dict(),
                save_eval_result = True,
                pre_net=pre_net
            )

        loss_sum = 0
        grad_norm = np.zeros(len(named_params))
