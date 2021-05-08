"""
1对1分类器 将会为每一对类别构造出一个分类器，
在预测阶段，收到最多投票的类别将会被挑选出来。
当存在结时（两个类具有同样的票数的时候）， 1对1分类器会选择总分类置信度最高的类，
其中总分类置信度是由下层的二元分类器 计算出的成对置信等级累加而成。

在one-vs-one策略中，同样假设有n个类别，则会针对两两类别建立二项分类器，得到k=n*(n-1)/2个分类器。对新数据进行分类时，依次使用这k个分类器进行分类，每次分类相当于一次投票，分类结果是哪个就相当于对哪个类投了一票。在使用全部k个分类器进行分类后，相当于进行了k次投票，选择得票最多的那个类作为最终分类结果​。

"""

from sklearn import datasets
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
iris = datasets.load_iris()
X, y = iris.data, iris.target
res = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)
print(res)