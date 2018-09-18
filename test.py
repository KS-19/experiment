#画像から角度回帰
import matplotlib
matplotlib.use('TkAgg')
import math

# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
# from sklearn.cross_validation import train_test_split　これ時代遅れ
from sklearn.model_selection import train_test_split # new
from PIL import Image

# train_x = test_x = np.array([[0,0], [0,1], [1,0], [1,1]])
# train_y = test_y = np.array([[0], [1], [1], [0]])
# train_n = train_x.shape[0]

# for i in range(5380):
#     p=i+1
#     im = np.array(Image.open('./image_gray/gray_%d.png' %p))
#     im = np.reshape(im, (1,64*64))
#     if(i==0):
#         image_list=im
#     else:
#         image_list = np.concatenate((image_list, im),axis=0)

# np.savetxt('gray_image.csv',image_list,delimiter=',')

image=np.loadtxt("gray_image.csv",delimiter=",")
# posture = np.loadtxt("posture_0710.csv",delimiter=",")
# posture = posture[:,1]
# a=np.linspace(0,56623,5380)
# b=np.round(a)
# c=np.array(list(map(int,b)))
# posture=posture[c]
# np.savetxt('posture_0710_2.csv',posture,delimiter=',')
posture=np.loadtxt("posture_0710_2.csv",delimiter=",")

label=np.array([])

for i in range(len(posture)):
    if(posture[i]>=0 and posture[i]<72):
        label = np.append(label,posture[i])
    elif(posture[i]>=72 and posture[i]<144):
        label = np.append(label,posture[i]-72)
    elif(posture[i]>=144 and posture[i]<216):
        label = np.append(label,posture[i]-144)
    elif(posture[i]>=216 and posture[i]<288):
        label = np.append(label,posture[i]-216)
    else:
        label = np.append(label,posture[i]-288)

mask = np.random.permutation(np.arange(len(posture)))
image_random=np.array([])
posture_random=np.array([])
for i in range(len(mask)):
    image_random=np.append(image_random,image[mask[i]])
    posture_random=np.append(posture_random,label[mask[i]])

    
image=np.reshape(image_random,(5380,4096))
posture=posture_random

posture=np.reshape(posture,(len(posture),1))
image = image/255
posture = posture/72

train_x, test_x, others_image = np.split(image,[4500,5000]) #3000x12288,1000x12288
train_y, test_y, others_label=np.split(posture,[4500,5000]) #3000x1,1000x1
train_n = train_x.shape[0]
test_n = test_x.shape[0]



# print(test_y.shape)


# class to do back prop
class Sigmoid:
    def __init__(self): # inatanceの生成には引数を必要としない
        self.y = None
        
    def __call__(self, x):
        y = 1 / (1 + np.exp(-x))   # 順伝播計算
        self.y = y
        return y
    
    def backward(self):
        return self.y * (1 -  self.y)  # 逆伝播計算

class ReLU:
    def __init__(self):
        self.x = None
        
    def __call__(self, x):
        self.x = x
        return x * (x > 0)  # 順伝播計算．これマスクを用いた計算になっていて，(x > 0)が真理値，Trueの際にxを返す機構になってる
    
    def backward(self):
        return 1 * (self.x > 0)  # 逆伝播計算

class Softmax:
    def __init__(self):
        self.y = None
        
    def __call__(self, x):
        exp_x = np.exp(x - x.max(axis=1, keepdims=True))  # ここで exp(x - x_max) を計算しよう
        y = exp_x / np.sum(exp_x, axis=1, keepdims=True) # exp_x を用いて softmax を計算しよう
        self.y = y
        return y

class Linear:
    def __init__(self, in_dim, out_dim, activation):
        self.W = np.random.uniform(low=-0.08, high=0.08, size=(in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.activation = activation()
        self.delta = None
        self.x = None
        self.dW = None
        self.db = None

    def __call__(self, x):
        # 順伝播計算
        self.x = x
        u = np.dot(x, self.W) + self.b  # self.W, self.b, x を用いて u を計算しよう
        self.z = self.activation(u)
        return self.z
    
    def backward(self, dout):
        # 誤差計算
        self.delta = dout * self.activation.backward() # dout と活性化関数の逆伝播 (self.activation.backward()) を用いて delta を計算しよう
        dout = np.dot(self.delta, self.W.T) # self.delta, self.W を用いて 出力 o を計算しよう
        
        # 勾配計算
        self.dW = np.dot(self.x.T, self.delta)  # dW を計算しよう
        self.db = np.dot(np.ones(len(self.x)), self.delta)  # db を計算しよう
        
        return dout

class MLP():
    def __init__(self, layers):
        self.layers = layers
        self.y = None
        
    def train(self, x, t, lr):     
        # 1. 順伝播
        self.y = x
        for layer in self.layers:
            self.y = layer(self.y)  # 順伝播計算を順番に行い， 出力 y を計算しよう
    
        # 2. 損失関数の計算
        # self.loss = np.sum(-t*np.log(self.y + 1e-7)) / len(x)
        # self.loss = np.sum(-t*np.log(self.y + 1e-7)) / len(x)
        self.loss = np.sum((self.y - t)**2/ (2 *len(x)))
                            
        # 3. 誤差逆伝播
        # 3.1. 最終層
        # 3.1.1. 最終層の誤差・勾配計算
        delta = (self.y - t) / len(self.layers[-1].x)
        self.layers[-1].delta = delta
        self.layers[-1].dW = np.dot(self.layers[-1].x.T, self.layers[-1].delta)
        self.layers[-1].db = np.dot(np.ones(len(self.layers[-1].x)), self.layers[-1].delta)
        dout = np.dot(self.layers[-1].delta, self.layers[-1].W.T)
        
        # 3.1.2. 最終層のパラメータ更新
        self.layers[-1].W -= lr * self.layers[-1].dW # self.layers[-1].dW を用いて最終層の重みを更新しよう
        self.layers[-1].b -= lr * self.layers[-1].db  # self.layers[-1].db を用いて最終層のバイアスを更新しよう
        
        # 3.2. 中間層
        for layer in self.layers[-2::-1]:
            # 3.2.1. 中間層の誤差・勾配計算
            dout = layer.backward(dout)  # 逆伝播計算を順番に実行しよう
            
            # 3.2.2. パラメータの更新
            layer.W -= lr * layer.dW  # 各層の重みを更新
            layer.b -= lr * layer.db  # 各層のバイアスを更新
            
        return self.loss

    def test(self, x, t):
        # 性能をテストデータで調べるために用いる
        # よって，誤差逆伝播は不要
        # 順伝播 (train関数と同様)
        self.y = x
        for layer in self.layers:
            self.y = layer(self.y)
        # self.loss = np.sum(-t*np.log(self.y + 1e-7)) / len(x)
        self.loss = np.sum( (t-self.y)**2 )/ (2 *len(x))
        return self.loss



# model = MLP([Linear(784, 1000, Sigmoid),Linear(1000, 1000, Sigmoid),Linear(1000, 10, Softmax)])
# model = MLP([Linear(784, 1000, ReLU),Linear(1000, 1000, ReLU),Linear(1000, 1000, ReLU),Linear(1000, 10, Softmax)])
model = MLP([Linear(4096, 100, Sigmoid), Linear(100, 1, Sigmoid)])


n_epoch = 2000
# epoch 4で0.9813をマーク
batchsize = 10
lr = 0.1
m=0.1

train_predict=np.array([])
train_label=np.array([])
loss_list=np.array([])


for epoch in range(n_epoch):
    # print('epoch %d | ' % epoch, end="")
    
    # 訓練
    sum_loss = 0
    pred_y = []
    perm = np.random.permutation(train_n)
    
    for i in range(0, train_n, batchsize):
        x = train_x[perm[i: i+batchsize]]
        t = train_y[perm[i: i+batchsize]]
        sum_loss += model.train(x, t, lr) * len(x)
        # model.y には， (N, 10)の形で，画像が0~9の各数字のどれに分類されるかの事後確率が入っている
        # そこで，最も大きい値をもつインデックスを取得することで，識別結果を得ることができる
        # pred_y.extend(np.argmax(model.y, axis=1))
    loss = sum_loss / train_n
    
    # accuracy : 予測結果を1-hot表現に変換し，正解との要素積の和を取ることで，正解数を計算できる．
    # accuracy = np.sum(np.eye(10)[pred_y] * train_y[perm]) / train_n
    # print('Train loss %.3f, accuracy %.4f | ' %(loss, accuracy), end="")
    train_predict = np.append(train_predict, model.y[0][0]*72)
    train_label = np.append(train_label,t[0]*72)
    loss_list = np.append(loss_list,loss)
    if epoch%100 == 0:
        print('Input', end='=')
        print(x, end='  Output=')        
        print(model.y*72, end=' label =')
        print(t*72, end=' MSE =')
        print(loss)

test_predict=np.array([])
for i in range(test_n):
    L=model.test(test_x[i],test_y[i])
    test_predict=np.append(test_predict,model.y*72)

x_axis=np.arange(test_n)
plt.rcParams["font.size"]=24
p1 =plt.plot(x_axis,test_predict,color='b') #blue
p2 =plt.plot(x_axis,test_y*72,color='r') #red
plt.legend([p1,p2], ["output", "label"])
plt.xlabel("test data")
plt.ylabel("posture")
#plt.plot(x_axis,loss_list*1000)
plt.show()
    
# x_axis=np.arange(10000)
# plt.plot(x_axis,train_predict)
# plt.plot(x_axis,train_label)
# plt.plot(x_axis,loss_list*10000)
# plt.show()

# y_x=np.arange(72)
# p1=plt.scatter(train_predict[1500:],train_label[1500:],color='b',s=50)
# p2=plt.scatter(test_predict,test_y*72,color='r',s=50)
# plt.legend([p1,p2], ["train", "test"])
# plt.xlabel("output")
# plt.ylabel("label")
# plt.plot(y_x,y_x,color='g')
# plt.show()

y_x=np.arange(72)
plt.scatter(train_predict[1500:],train_label[1500:],color='b',s=50)
#plt.scatter(test_predict,test_y*72,color='r',s=50)
#plt.legend([p1,p2], ["train", "test"])
plt.title("train")
#plt.legend("train")
plt.xlabel("output")
plt.ylabel("label")
plt.plot(y_x,y_x,color='g')
plt.show()

y_x=np.arange(72)
#plt.scatter(train_predict[1500:],train_label[1500:],color='b',s=50)
plt.scatter(test_predict,test_y*72,color='r',s=50)
#plt.legend([p1,p2], ["train", "test"])
plt.title("test")
#plt.legend("test")
plt.xlabel("output")
plt.ylabel("label")
plt.plot(y_x,y_x,color='g')
plt.show()

# loss= model.test(test_x,test_y)
# print(loss/test_n)
# x_axis=np.arange(test_n) 
# plt.plot(x_axis,model.y*360)
# plt.plot(x_axis,test_y*360)
# plt.show()

