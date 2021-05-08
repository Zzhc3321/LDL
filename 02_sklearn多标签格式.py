# np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0]]) 表示第一个样本属于第 0 个标签，第二个样本属于第一个和第二个标签，第三个样本不属于任何标签。

from sklearn.preprocessing import MultiLabelBinarizer

y = [[2, 3, 4], [2], [0, 1, 3], [0, 1, 2, 3, 4], [0, 1, 2]]

res = MultiLabelBinarizer().fit_transform(y)#one-hot编码

print(res)