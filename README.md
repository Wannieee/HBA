# HBA
*Fast mmwave Beam Alignment via Correlated Bandit Learning*  
https://ieeexplore.ieee.org/document/8842625  
## 1 integrated_channel_and_test.py
### 1.1 channel_rss
函数channel_rss用于计算给定移相权重下的输出的RSS，包括Mean_RSS和Noisy_RSS  
函数体中在计算Mean_RSS时可选两种计算方案，一是将路损g单位为倍数的情况下求平均，二是路损g单位为db的情况下求平均
### 1.2 main
用于测试函数channel_rss的效果，输出在相同信道条件下，使用不同波束可以得到的增益  
## 2 test.py
### 2.1 node_to_beam_idx
HBA算法本身在执行前就已经内定了一颗完美二叉树，在执行时还会生成一颗T树和P树。  
T树保存所有加入的节点（当然这些节点来自完美二叉树）  
P树则保存一次循环中从根节点到叶节点再到新加入节点的路径  
程序执行时需要把节点保存在T树里面，但还需要创建一个使用到T树节点的P树  
这两颗树共用一些节点，可以直接在定义Node的时候再加上来两个分支代表P树中的左右节点，但这样会显得麻烦  
所以采用一个列表p记录一次循环是如何从根节点到叶节点和新节点中的，比如p=[0,1,0]代表head_node.left_child.right_child.left_child  
即P树中包含了head_node、head_node.left_child、head_node.left_child.right_child、head_node.left_child.right_child.left_child  
一方面需要P树来更新其中每个节点的R值  
另一方面需要P树指代一次循环得到的区域，这便是由函数node_to_beam_idx来计算，而还需要m来决定波束的数目  
区域到波束域的映射也略有不同，比如m=7的情况，即k=128时：  
波束空间可以设为[0,128]，通过p可以得到该区间的一个子区间，然后找到子区间正中的那个点，再找到离该点最近的整数作为所选择的波束  
这样波束空间中选择0和128得到的是同一个波束，并且如果点是随机选的，那么所有波束都是等可能地选到  
### 2.2 beam_gain_map
将一个rss数值转换为一个reward数值，不能对列表进行处理，是个可以进一步改进的地方  
上下限根据integrated_channel_and_test.py中main函数得到的图进行粗略地选择  
### 2.3 Node
创建了一个树节点的类，更新E值的主要参数也在类里面初始化定义  
更新Q值时必须采用后序遍历，更新E值时其实怎么遍历法都可以  
R、E、Q值的更新和树的遍历也都封装到类里面了  
在更新E值时可以稍作更改而变为HOO算法  
### 2.4 main
把HBA的整个过程复现了一下，先定义深度m=7，才得到天线数和波束数n=k=2^7=128  
当到达深度时停止循环，将最后一次循环得到的p对应的中间波束作为最终选择  
其实这是存在问题的，因为最后一次循环是头一次到深度m，即这个新节点是深度为m-1的节点随机选择左右子树而创建的  
因此这儿存在两个问题  
一是HBA整个过程的封装  
二是最后如何终止循环并返回最优beam  
## 3 integrated_simulation
### simulation
定义了一个函数simulation，输入初始条件返回每个time slot中所采用的波束、得到的rss、累积遗憾和最终选择的波束  
选择不同方法也是返回这些值，这里只添加了一种HBA的方法  
HBA最终的波束选择：  
稍微改进了一下，当最后一次循环得到了p，实际上代表的是p.pop()后p所代表的区域包含了最优波束，由于len(p)正好是深度m  
在[0,128]中，p指代的区域正好夹在两个相邻整数中间，p.pop()后指代的区域为其两倍，需要在该区域中遍历搜索找到最优波束  
p.pop后，p指代区域正中的波束在某次循环中一定被测过，得到该区域的其它两个波束，这两个波束有可能也被测过  
于是这三个波束应该全做一次查找，再决定这一时刻和下一时刻测哪个波束或是终止BA  
再输出时，把向量用python自带的列表表示，因此有个tolist()操作  
### main
稍微测试了一下，再改进E的更新后可以在30步以内收敛了，但HBA和HOO的性能差异好像不明显
test_stability设为1测试稳定性，采用同一个angle进行测试
test_generalization设为1测试泛化性，每次采用随机angle进行测试
对于稳定性，测试发现收敛到LOS的概率随角度变化不稳定
对于泛化性，测试发现要收敛到LOS的概率在70%左右
## 4 origin_version
将前面3个代码的内容放在了一起
## 5 improve_version
在origin_version基础上做后续改进
已有的改进：
1、可以重新自定义波束码本