#coding:utf-8
from load_data import get_data_loader,get_data
from model_base import GATModel
import argparse
import os
from torch.optim import AdamW,Adagrad
import torch
from sklearn.metrics import accuracy_score
from data_plot import plot_scatter
import logging

logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                    filename='log.txt',
                    filemode='w',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    # format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'#日志格式
                    )


device="cuda" if torch.cuda.is_available() else "cpu"
print("device:",device)

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--do_train', action='store_true',default=True)
    parser.add_argument('--do_valid', action='store_true',default=True)
    parser.add_argument('--do_test', action='store_true',default=True)
    parser.add_argument('--train_data', type=str, default="../datapkls/liao_train_200529.pkl" )
    parser.add_argument('--valid_data', type=str, default="../datapkls/liao_val_200529.pkl" )
    parser.add_argument('--test_data', type=str, default="../datapkls/liao_test_200529.pkl" )

    parser.add_argument('--input_dim', default=25626, type=int)
    parser.add_argument('--hidden_dim', default=8, type=int)
    parser.add_argument('--nHeads', default=4, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_label',default=3,type=int)
    parser.add_argument('--number_parts',default=5000,type=int)


    parser.add_argument('-lr', '--learning_rate', default=0.0003, type=float)
    parser.add_argument('-wd', '--weight_decay', default=5e-2, type=float)
    parser.add_argument('-init', '--init_checkpoint', default="weights", type=str)
    # parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default="weights", type=str)
    parser.add_argument('--max_epoch', default=30, type=int)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    return parser.parse_args(args)

def evaluate(model, valid_data, valid_cl):
    model.eval()
    preds=[]
    labels=[]
    losses=[]
    for batch_data in valid_cl:
        label=batch_data.y
        labels.extend(label)
        batch_data.to(device)
        output,loss,_=model(batch_data)
        losses.append(loss.item())
        pred=torch.argmax(output,dim=1)
        preds.extend(list(pred.detach().cpu().numpy()))
    acc=accuracy_score(labels,preds)
    losses=sum(losses)/len(losses)
    return acc,losses







def main(args):
    model=GATModel(input_dim=args.input_dim,hidden_dim=args.hidden_dim,num_label=args.num_label,nHeads=args.nHeads)
    model.to(device)
    optimizer=AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    # optimizer=Adagrad(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)

    train_data=get_data(args.train_data)
    train_cl=get_data_loader(train_data,number_parts=args.number_parts,batch_size=args.batch_size)
    valid_data=get_data(args.valid_data)
    valid_cl=get_data_loader(valid_data,number_parts=args.number_parts,batch_size=args.batch_size)

    test_data=get_data(args.test_data)
    test_cl=get_data_loader(test_data,number_parts=args.number_parts,batch_size=args.batch_size)

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

                    torch.save(model, "weights/covid19_patients_capsule.pt", _use_new_zipfile_serialization=False)
    else:
        import numpy as np
        import pickle
        datapkl = pickle.load(open(args.test_data,"rb"))
        condition=datapkl["Condition"]
        ycondition=datapkl["ycondition"]
        y2label={y:label for y,label in zip(ycondition,condition)}
        label_set=None

        model=torch.load("weights/covid19_patients_capsule.pt")
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
        vectors=np.array(vectors)
        # vectors=datapkl["X"]
        plot_scatter(vectors,labels,label_set)



if __name__=="__main__":
    args=parse_args()
    main(args)



