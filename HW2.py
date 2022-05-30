import numpy as np
from datetime import datetime      # datetime.now() 를 이용하여 학습 경과 시간 측정

def sigmoid(x):         # sigmoid 함수
    return 1 / (1+np.exp(-x))

def cross_entropy(t, y) :
    delta = 1e-7    # log 무한대 발산 방지
    return -np.sum(t*np.log(y+delta) + (1-t)*np.log((1-y)+delta))    

def numerical_derivative(f, x):      # 수치미분 함수
    delta_x = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index        
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x) # f(x+delta_x)
        
        x[idx] = tmp_val - delta_x 
        fx2 = f(x) # f(x-delta_x)
        grad[idx] = (fx1 - fx2) / (2*delta_x)
        
        x[idx] = tmp_val 
        it.iternext()   
    return grad

class LogicGate:
    def __init__(self, gate_name, xdata, tdata):
        self.name = gate_name
        # 입력 데이터, 정답 데이터 초기화
        self.xdata = xdata.reshape(4,2)  # 4개의 입력데이터 x1, x2 에 대하여 batch 처리 행렬
        self.tdata = tdata.reshape(4,1)  # 4개의 입력데이터 x1, x2 에 대한 각각의 계산 값 행렬
        
        # 2층 hidden layer unit : 3개 가정,  가중치 W2, 바이어스 b2 초기화
        self.W2 = np.random.rand(2,3)  # weight, 2 X 3 matrix
        self.b2 = np.random.rand(3)
        
        # 3층 hidden layer unit : 2 개 , 가중치 W3, 바이어스 b3 초기화
        self.W3 = np.random.rand(3,2)  # weight, 3 X 2 matrix
        self.b3 = np.random.rand(2)

        # 4층 output layer unit : 1 개 , 가중치 W4, 바이어스 b4 초기화
        self.W4 = np.random.rand(2,1)  # weight, 2 X 1 matrix
        self.b4 = np.random.rand(1)
                        
        # 학습률 learning rate 초기화
        self.lr = 1e-2
        print(self.name + " object is created")
            
    def feed_forward(self):        # errFunc()함수 대신 feed forward를 통하여 손실함수(cross-entropy) 값 계산
        z2 = np.dot(self.xdata, self.W2) + self.b2  # 은닉층의 선형회귀 값
        a2 = sigmoid(z2)                            # 은닉층의 출력
        z3 = np.dot(a2, self.W3) + self.b3          # 은닉층의 선형회귀 값
        a3 = sigmoid(z3)                            # 은닉층의 출력
        z4 = np.dot(a3, self.W4) + self.b4          # 출력층의 선형회귀 값
        y = a4 = sigmoid(z4)                        # 출력층의 출력
        return cross_entropy(self.tdata, y)         # 출력의 손실값 리턴
    
    def errValue(self):             # 외부 출력을 위한 손실함수(cross-entropy) 값 계산 
        return  self.feed_forward()
    
    def train(self):            # 수치미분을 이용하여 손실함수가 최소가 될때 까지 학습하는 함수
        f = lambda x : self.feed_forward()
        start = datetime.now()
        for step in range(55001):
            self.W2 -= self.lr * numerical_derivative(f, self.W2)
            self.b2 -= self.lr * numerical_derivative(f, self.b2)
            self.W3 -= self.lr * numerical_derivative(f, self.W3)
            self.b3 -= self.lr * numerical_derivative(f, self.b3)
            self.W4 -= self.lr * numerical_derivative(f, self.W4)
            self.b4 -= self.lr * numerical_derivative(f, self.b4)
            if (step % 2000 == 0):
                print("Step = {:<5d}\tError Val = {:.4f}".format(step, self.errValue()))
        print("Training time = ", datetime.now() - start)

    def predict(self, test):      # query, 즉 미래 값 예측 함수
        z2 = np.dot(test, self.W2) + self.b2         # 은닉층의 선형회귀 값
        a2 = sigmoid(z2)                             # 은닉층의 출력
        z3 = np.dot(a2, self.W3) + self.b3         # 은닉층의 선형회귀 값
        a3 = sigmoid(z3)                             # 은닉층의 출력
        z4 = np.dot(a3, self.W4) + self.b4           # 출력층의 선형회귀 값
        y = a4 = sigmoid(z4)                         # 출력층의 출력
        if y > 0.5:
            result = 1  # True
        else:
            result = 0  # False
        return result, y
      
 # XOR Gate 객체 생성
xdata = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
tdata = np.array([0, 1, 1, 0])

xor = LogicGate("XOR", xdata, tdata)
xor.train() 

test_data = np.array([ [0, 0], [0, 1], [1, 0], [1, 1], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5], [0.4, 0.6], [0.3, 0.7] ])
for data in test_data:
    r, y = xor.predict(data)
    print(data, "-->", r, "[%.3f]"%y[0])
