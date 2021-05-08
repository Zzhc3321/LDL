
"""
在one-vs-all策略中，假设有n个类别，那么就会建立n个二项分类器，每个分类器针对其中一个类别和剩余类别进行分类。进行预测时，利用这n个二项分类器进行分类，得到数据属于当前类的概率，选择其中概率最大的一个类别作为最终的预测结果。

这个方法在于每一个类都将用一个分类器进行拟合。
对于每一个分类器，该类将会和其他所有的类有所区别。
除了它的计算效率之外 (只需要 n_classes 个分类器), 这种方法的优点是它具有可解释性。
因为每一个类都可以通过有且仅有一个分类器来代表，所以通过检查一个类相关的分类器就可以获得该类的信息。
这是最常用的方法，也是一个合理的默认选择。
"""

from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
iris = datasets.load_iris()
X, y = iris.data, iris.target
res = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)

print(res)