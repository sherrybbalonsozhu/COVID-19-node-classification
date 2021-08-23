#coding:utf-8
from load_data import get_data_loader,get_data
from model_capsule import CapsuleGNNModel
import argparse
import os
from torch.optim import AdamW,Adagrad,Adam
import torch
from sklearn.metrics import accuracy_score
from data_plot import plot_scatter,plot_feature
import logging
from collections import defaultdict

logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                    filename='log.txt',
                    filemode='w',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    # format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'#日志格式
                    )

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device="cuda" if torch.cuda.is_available() else "cpu"
print("device:",device)
# device="cpu"

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--do_train', action='store_true',default=False)
    parser.add_argument('--do_valid', action='store_true',default=True)
    parser.add_argument('--do_test', action='store_true',default=True)
    parser.add_argument('--train_data', type=str, default="../datapkls/hbec_train_200529.pkl" )
    parser.add_argument('--valid_data', type=str, default="../datapkls/hbec_val_200529.pkl" )
    parser.add_argument('--test_data', type=str, default="../datapkls/hbec_test_200529.pkl" )

    parser.add_argument('--input_dim', default=24714, type=int)
    parser.add_argument('--hidden_dim', default=16, type=int) #64
    parser.add_argument('--nHeads', default=2, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_label',default=7,type=int)
    parser.add_argument('--number_parts',default=10000,type=int)



    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
    parser.add_argument('-wd', '--weight_decay', default=1e-4, type=float)
    parser.add_argument('-init', '--init_checkpoint', default="weights", type=str)
    # parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default="weights", type=str)
    parser.add_argument('--max_epoch', default=50, type=int)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    return parser.parse_args(args)

def evaluate(model, valid_data, valid_cl,y2label=None):
    model.eval()
    preds=[]
    labels=[]
    losses=[]
    for batch_data in valid_cl:
        label=batch_data.y
        labels.extend(label.numpy())
        batch_data.to(device)
        output,loss,_=model(batch_data)
        losses.append(loss.item())
        pred=torch.argmax(output,dim=1)
        preds.extend(list(pred.detach().cpu().numpy()))
    acc=accuracy_score(labels,preds)
    label2rights=defaultdict(list)
    for label,pred in zip(labels,preds):
        if label==pred:
            label2rights[label].append(1)
        else:
            label2rights[label].append(0)
    for label,rights in label2rights.items():
        label_acc=sum(rights)/len(rights)
        if y2label is not None:
            print(label,y2label[label],len(rights),label_acc)
        else:
            print(label, len(rights), label_acc)

    losses=sum(losses)/len(losses)
    return acc,losses







def main(args):
    model=CapsuleGNNModel(input_dim=args.input_dim,hidden_dim=args.hidden_dim,num_label=args.num_label,nHeads=args.nHeads)
    model.to(device)
    optimizer=Adam(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    # optimizer=AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    # optimizer=Adagrad(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)

    train_data=get_data(args.train_data,label="yinftime")
    train_cl=get_data_loader(train_data,number_parts=args.number_parts,batch_size=args.batch_size)
    valid_data=get_data(args.valid_data,label="yinftime")
    valid_cl=get_data_loader(valid_data,number_parts=args.number_parts,batch_size=args.batch_size,shuffle=False)

    test_data=get_data(args.test_data,label="yinftime")
    test_cl=get_data_loader(test_data,number_parts=args.number_parts,batch_size=args.batch_size,shuffle=False)

    if args.do_train:
        model.train()
        optimizer.zero_grad()

        for epoch in range(args.max_epoch):
            losses = []
            for batch_data in train_cl:
                batch_data=batch_data.to(device)
                output,loss,_=model(batch_data)

                losses.append(loss.item())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            losses = sum(losses) / len(losses)
            # print("epoch: %s, loss: %s" % (epoch, losses))
            if epoch % 1 == 0 or epoch == args.max_epoch - 1:
                with torch.no_grad():
                    model.eval()
                    train_acc, train_loss = evaluate(model, train_data, train_cl)

                    acc,valid_loss = evaluate(model, valid_data, valid_cl)
                    print("epoch: %s, train_loss: %s, train_acc: %s valid_loss: %s, valid acc: %s"%(epoch,train_loss,train_acc,valid_loss,acc))
                    logging.info(acc)

                    test_acc,test_loss = evaluate(model, test_data, test_cl)
                    print("epoch: %s, test_loss: %s, test_acc: %s"%(epoch,test_loss,test_acc))

                    torch.save(model,"weights/hbec_capsule.pt",_use_new_zipfile_serialization=False)
    else:
        import numpy as np
        import pickle
        datapkl = pickle.load(open(args.test_data,"rb"))
        inf_timepoint=datapkl["inf_timepoint"]
        yinftime=datapkl["yinftime"]
        y2label={y:label for y,label in zip(yinftime,inf_timepoint)}
        label_set=["0_1dpi","0_2dpi","0_3dpi","1_1dpi","1_2dpi","1_3dpi","0_Mock"]

        model=torch.load("weights/hbec_capsule.pt")
        test_acc, test_loss = evaluate(model, test_data, test_cl)
        print("epoch: %s, test_loss: %s, test_acc: %s" % (0, test_loss, test_acc))

        vectors=[]
        labels=[]
        for batch_data in test_cl:
            label = batch_data.y
            labels.extend(list(label.numpy()))

            batch_data.to(device)
            output,loss,features=model(batch_data)
            output=list(features.detach().cpu().numpy())
            vectors.extend(output)
        # labels=datapkl["yinftime"]
        labels=[y2label[y] for y in labels]
        evaluate(model,test_data,test_cl,y2label)
        vectors=np.array(vectors)
        # vectors=datapkl["X"]
        # plot_scatter(vectors,labels,label_set)

        plot_feature(model,nheads=args.nHeads,n_hidden=args.hidden_dim)





if __name__=="__main__":
    args=parse_args()
    main(args)



