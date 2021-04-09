# dmHomework1
数据挖掘的第一次互评作业。

本次作业使用的数据集为Wine Reviews与Chicago Building Violations。

**【由于频数统计、五数概括文件过多，因此没有在报告中详细介绍，详情请见文件夹内】**
## 使用说明：
### 数据集：
数据集应该放在根目录的data文件夹下。
### 代码：
对数据集Wine Reviews与Chicago Building Violations进行处理的python文件分别为wine_process.py和building_process.py，运行时放在根目录下。
### 结果：
nominal文件夹下存放的是标称属性的频数统计结果，以txt文件保存；

numberical文件夹下存放的是数值属性的五数概括和缺失值（missingValues.txt），以txt文件保存，以及数值属性的直方图以及盒图；

delete文件夹下存放的是**将缺失部分剔除**后的可视化结果；

mod文件夹下存放的是**用最高频率值来填补缺失值**后的可视化结果；

related文件夹下存放的是**通过属性的相关关系来填补缺失值**后的可视化结果；

similarity文件夹下存放的是**通过数据对象之间的相似性来填补缺失值**后的可视化结果。
