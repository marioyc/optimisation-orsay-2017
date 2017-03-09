import scipy.io

def load_data():
    mat = scipy.io.loadmat('data_orsay_2017.mat')
    Xtrain = mat['Xtrain']
    Xtest = mat['Xtest']
    ytrain = mat['ytrain']
    ytest = mat['ytest']
    return Xtrain, ytrain, Xtest, ytest

if __name__ == '__main__':
    Xtrain, ytrain, Xtest, ytest = load_data()
    print Xtrain.shape
    print ytrain.shape
