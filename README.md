# dmHomework1
数据挖掘的第一次互评作业
本次作业使用的数据集为Wine Reviews与Chicago Building Violations。
## 使用说明：
### 数据集：
数据应该放在data文件夹下。
### 代码：
对应数据集的代码已放到各自的文件夹下，Wine Reviews对应wine/process.py，Chicago Building Violations对应building/process.py。使用时将python文件放到与data文件夹同一目录下运行，或者自主修改代码中数据集的路径即可。
### 结果：
nominal文件夹下存放的是标称属性的频数统计结果，以txt文件保存；

numberical文件夹下存放的是数值属性的五数概括，以txt文件保存；

delete文件夹下存放的是**将缺失部分剔除**后的可视化结果；

mod文件夹下存放的是**用最高频率值来填补缺失值**后的可视化结果；

related文件夹下存放的是**通过属性的相关关系来填补缺失值**后的可视化结果；

similarity文件夹下存放的是**通过数据对象之间的相似性来填补缺失值**后的可视化结果。
