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
for r in range(5,8):
    param.Round = r
    print(param.Round,' rounds:')
    for i in range(1):
        X_eval, Y_eval = get_eval_data(n=10**6, nr=param.Round, group_size=param.group_size)    
        model = load_model("./"+param.diffstr+"/"+str(param.Round)+"_distinguisher.h5")
        evaluate(model, X_eval, Y_eval)



# Evaluation of neural distinguisher constructed with method 1
print("Evaluation of neural distinguisher constructed with method 1")
for r in range(5,8):
    param.Round = r
    print(param.Round,' rounds:')
    for i in range(1):
        X_eval, Y_eval = get_eval_data_filter1(n=10**6, nr=param.Round, group_size=param.group_size)
        model = load_model("./"+param.diffstr+"/filter/"+str(param.Round)+"_distinguisher.h5")
        evaluate(model, X_eval, Y_eval)
        

# Evaluation of neural distinguisher constructed with method 1
print("Evaluation of neural distinguisher constructed with method 2")
for r in range(8,9):
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
5  rounds:
Accuracy:  0.907313 TPR:  0.86969 TNR:  0.944936 MSE: 0.06938087031982736
6  rounds:
Accuracy:  0.754492 TPR:  0.672648 TNR:  0.836336 MSE: 0.16373340178304477
7  rounds:
Accuracy:  0.587049 TPR:  0.539694 TNR:  0.634404 MSE: 0.23777964346027763
Evaluation of neural distinguisher constructed with method 1
5  rounds:
Accuracy:  0.9958018758748388 TPR:  0.9952926512928537 TNR:  0.9962705679658298 MSE: 0.0030643681166829793
6  rounds:
Accuracy:  0.992306101837788 TPR:  0.9913655460310947 TNR:  0.9930625802609963 MSE: 0.005591664468017485
7  rounds:
Accuracy:  0.9784379157446823 TPR:  0.9774391509294121 TNR:  0.9792875810605893 MSE: 0.01565486208718064
Evaluation of neural distinguisher constructed with method 2
8  rounds, s = 1:
Accuracy:  0.5545050749492847 TPR:  0.23472981087873554 TNR:  0.8280604398217881 MSE: 0.24504842
8  rounds, s = 2:
Accuracy:  0.5532081958341734 TPR:  0.0 TNR:  1.0 MSE: 0.24716882


'''