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

wdir = './data/'
def get_eval_data(n, nr,group_size=1):
    num = n // 2
    assert num % group_size == 0
    
    X_p = np.load(wdir+str(int(log10(n)))+"_Xp_"+str(nr)+".npy")
    X_n = np.load(wdir+str(int(log10(n)))+"_Xn_"+str(nr)+".npy")
    Y_p = [1 for i in range(num // group_size)]
    Y_n = [0 for i in range(num // group_size)]
    X = np.concatenate((X_p, X_n), axis=0).reshape(n // group_size, -1)
    Y = np.concatenate((Y_p, Y_n))
    return X, Y

def get_eval_data_filter1(n, nr, group_size=1):
    num = n // 2
    assert num % group_size == 0
    
    from math import log10
    X_p = np.load(wdir+str(int(log10(n)))+"_Xp_filter_"+str(nr)+".npy")
    X_n = np.load(wdir+str(int(log10(n)))+"_Xn_filter_"+str(nr)+".npy")
    lp = (len(X_p) // group_size)
    ln = (len(X_n) // group_size)

    X_p = X_p[:lp*group_size]
    X_n = X_n[:ln*group_size]
    Y_p = [1 for i in range(lp)]
    Y_n = [0 for i in range(ln)]
    X = np.concatenate((X_p, X_n), axis=0).reshape((lp + ln) // group_size, -1)
    Y = np.concatenate((Y_p, Y_n))
    return X, Y


def get_eval_data_filter2(n,r_mid,r_end):
    
    new_ctdata0l = np.load(wdir+str(n)+ "_" + str(r_mid)+ "_" + str(r_end)+"_new_ctdata0l.npy")
    new_ctdata0r = np.load(wdir+str(n)+ "_" + str(r_mid)+ "_" + str(r_end)+"_new_ctdata0r.npy")
    new_ctdata1l = np.load(wdir+str(n)+ "_" + str(r_mid)+ "_" + str(r_end)+"_new_ctdata1l.npy")
    new_ctdata1r = np.load(wdir+str(n)+ "_" + str(r_mid)+ "_" + str(r_end)+"_new_ctdata1r.npy")
    Y = np.load(wdir+str(n)+ "_" + str(r_mid)+ "_" + str(r_end)+"_new_Y.npy")
    X = convert_to_binary_new(np.array(np.array((new_ctdata0l, new_ctdata0r)) ^ np.array((new_ctdata1l, new_ctdata1r))),16,2);

    return X, Y



# Evaluating the neural distinguisher with the differences
print("Evaluating the neural distinguisher with the differences")
for r in range(6,10):
    param.Round = r
    print(param.Round,' rounds:')
    for i in range(1):
        X_eval, Y_eval = get_eval_data(n=10**6, nr=param.Round, group_size=param.group_size)    
        model = load_model("./"+param.diffstr+"/"+str(param.Round)+"_distinguisher.h5")
        evaluate(model, X_eval, Y_eval)



# Evaluation of neural distinguisher constructed with method 1
print("Evaluation of neural distinguisher constructed with method 1")
for r in range(7,10):
    param.Round = r
    print(param.Round,' rounds:')
    for i in range(1):
        X_eval, Y_eval = get_eval_data_filter1(n=10**6, nr=param.Round, group_size=param.group_size)
        model = load_model("./"+param.diffstr+"/filter/"+str(param.Round)+"_distinguisher.h5")
        evaluate(model, X_eval, Y_eval)
        

# Evaluation of neural distinguisher constructed with method 1
print("Evaluation of neural distinguisher constructed with method 2")
for r in range(10,11):
    param.Round = r
    for i in range(1):
        for s in range(1,3):
            print(param.Round,' rounds, s = '+str(s)+':')
            r_mid = param.Round-s
            X_eval, Y_eval = get_eval_data_filter2(n=10**6, r_mid = r_mid, r_end = param.Round)
            model = load_model("./"+param.diffstr+"/filter/"+str(r_mid)+"_"+str(param.Round)+"_distinguisher.h5")
            evaluate(model, X_eval, Y_eval);

'''
Evaluating the neural distinguisher with the differences
6  rounds:
Accuracy:  0.990649 TPR:  0.999126 TNR:  0.982172 MSE: 0.008195390878451007
7  rounds:
Accuracy:  0.837517 TPR:  0.849156 TNR:  0.825878 MSE: 0.1093568018007628
8  rounds:
Accuracy:  0.656257 TPR:  0.562926 TNR:  0.749588 MSE: 0.2116535150892919
9  rounds:
Accuracy:  0.564904 TPR:  0.46224 TNR:  0.667568 MSE: 0.24302461384029042
Evaluation of neural distinguisher constructed with method 1
7  rounds:
Accuracy:  0.9865888008367671 TPR:  0.9869966225605419 TNR:  0.9861694682110882 MSE: 0.009869766875642956
8  rounds:
Accuracy:  0.9808689212540114 TPR:  0.9764338474328775 TNR:  0.9841995309485885 MSE: 0.0140683706934164
9  rounds:
Accuracy:  0.9797360930153515 TPR:  0.9762531154804763 TNR:  0.9821475649506279 MSE: 0.014783587695986991
Evaluation of neural distinguisher constructed with method 2
10  rounds, s = 1:
Accuracy:  0.6439174706674667 TPR:  0.34891006114027695 TNR:  0.8486083129467069 MSE: 0.215176
10  rounds, s = 2:
Accuracy:  0.5776734060045065 TPR:  0.11159499376346524 TNR:  0.9287674304384036 MSE: 0.24164535
'''