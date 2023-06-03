import numpy as np
import param
from math import log10


from keras.models import model_from_json,load_model




def convert_to_binary_new(arr,WORD_SIZE=16,NO_OF_WORDS=2):
  X = np.zeros((NO_OF_WORDS * WORD_SIZE,len(arr[0])),dtype=np.uint8);
  for i in range(NO_OF_WORDS * WORD_SIZE):
    index = i // WORD_SIZE;
    offset = WORD_SIZE - (i % WORD_SIZE) - 1;
    X[i] = (arr[index] >> offset) & 1;
  X = X.transpose();
  return(X);

def evaluate(net,X,Y):
    Z = net.predict(X,verbose=0,batch_size=10000).flatten();
    Zbin = (Z > 0.5);
    diff = Y - Z; mse = np.mean(diff*diff);
    n = len(Z); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    acc = np.sum(Zbin == Y) / n;
    tpr = np.sum(Zbin[Y==1]) / n1;
    tnr = np.sum(Zbin[Y==0] == 0) / n0;
    mreal = np.median(Z[Y==1]);
    high_random = np.sum(Z[Y==0] > mreal) / n0;
    print("Accuracy: ", acc, "TPR: ", tpr, "TNR: ", tnr, "MSE:", mse);
    # print("Percentage of random pairs with score higher than median of real pairs:", 100*high_random);



def get_eval_data_filter2(n,r_mid,r_end):
    wdir = './data/'
    new_ctdata0l = np.load(wdir+str(n)+ "_" + str(r_mid)+ "_" + str(r_end)+"_new_ctdata0l.npy")
    new_ctdata0r = np.load(wdir+str(n)+ "_" + str(r_mid)+ "_" + str(r_end)+"_new_ctdata0r.npy")
    new_ctdata1l = np.load(wdir+str(n)+ "_" + str(r_mid)+ "_" + str(r_end)+"_new_ctdata1l.npy")
    new_ctdata1r = np.load(wdir+str(n)+ "_" + str(r_mid)+ "_" + str(r_end)+"_new_ctdata1r.npy")
    Y = np.load(wdir+str(n)+ "_" + str(r_mid)+ "_" + str(r_end)+"_new_Y.npy")
    X = convert_to_binary_new(np.array(np.array((new_ctdata0l, new_ctdata0r)) ^ np.array((new_ctdata1l, new_ctdata1r))),16,2);

    return X, Y



# Evaluation of neural distinguisher constructed with method 2
print("Evaluation of neural distinguisher constructed with method 2")
for r in range(8,9):
    param.Round = r
    for i in range(1):
        for s in range(1,3):
            print(param.Round,' rounds, s = '+str(s)+':')
            r_mid = param.Round-s
            X_eval, Y_eval = get_eval_data_filter2(n=10**6, r_mid = r_mid, r_end = param.Round)
            model = load_model("./"+param.diffstr+"/best"+str(r_mid)+"_"+str(param.Round)+"depth2.h5")
            evaluate(model, X_eval, Y_eval);

'''
Evaluation of neural distinguisher constructed with method 2
8  rounds, s = 1:
Accuracy:  0.5545050749492847 TPR:  0.23472981087873554 TNR:  0.8280604398217881 MSE: 0.24504845
8  rounds, s = 2:
Accuracy:  0.5532081958341734 TPR:  0.0 TNR:  1.0 MSE: 0.24716882
'''