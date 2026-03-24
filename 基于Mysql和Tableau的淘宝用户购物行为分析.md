# 淘宝用户购物行为数据分析
## 一、项目背景和目标
1.背景：基于淘宝用户行为数据挖掘用户行为模式、商品偏好及平台运营问题

2.目标：
* 掌握电商领域核心指标体系（UV/PV、转化率、复购率等）
* 挖掘用户行为规律与商品特征
* 输出可落地的业务优化建议

## 二、工具与技术栈
* **数据存储与处理：** MySQL (数据清洗、SQL分析)
* **数据分析：** Python (Pandas/Numpy/Matplotlib...)
* **可视化：** Tableau (交互式仪表盘)

## 三、项目流程与内容
## 阶段1：数据理解与清洗
### 数据获取与探索
* 数据来源：https://tianchi.aliyun.com/dataset/649
* 数据集介绍：本数据集包含了2017年11月25日至2017年12月3日之间，有行为的约一百万随机用户的所有行为（行为包括点击、购买、加购、喜欢）。
* 字段理解：用户ID、商品ID、商品类目、行为类型(pv/buy/cart/fav)、时间戳  
![字段解释](images\prepare_python\字段解释.png)
* 数据量级统计  
 用户数量987994，商品数量4162024，用户数量987994，商品类目数量9439，所有行为数量100150807。     
    * 为什么这么做？  
    **1. 判断数据是否完整：** 例如期望用户100万，实际只有10万，说明数据缺失。  
    **2.数据规模是否合理：** 例如用户有100万，行为有一亿条，平均每个用户100条行为  
    **3.判断是否需要大数据处理** 例如几万数据用excel处理，几百万数据用Pandas处理，几亿数据用Spark处理。  
```python
# 数据预览
import pandas as pd

# 加载数据
df=pd.read_csv('data\UserBehavior.csv',names=['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp'])

# 查看前五行：确认数据是否读取成功，检查字段顺序是否正确，观察数据格式
df=pd.read_csv('data\UserBehavior.csv',names=['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp'],nrows=5)
```

![查看前五行](images\prepare_python\head.png)

```python
# 基础统计信息
df.info() #数据类型
```
![查看数据结构](images\prepare_python\df_info.png)

```python
df.isnull().sum() #查看缺失值数量
```
![查看缺失值数量](images\prepare_python\isnull_sum.png)

```python
print(df.describe()) #数值字段的统计信息,用于发现异常数据
```
![数值字段的统计信息](images\prepare_python\dfdescribe.png)

```python
print(df['behavior_type'].value_counts()) #行为类型统计，为后面分析提供依据
```
![行为类型统计](images\prepare_python\behaviorType_valueCounts.png)

```python
# 转换为datetime格式
import datetime
df['datetime']=pd.to_datetime(df['timestamp'],unit='s')
print(df['datetime'].head())
```
![转换为datetime格式](images\prepare_python\to_datetime.png)

```python
# 提取日期、小时、星期几
df['date']=df['datetime'].dt.date
df['hour']=df['datetime'].dt.hour
df['weekday']=df['datetime'].dt.weekday #0=周一, 6=周日
print(df['date'][:5] )
print(df['hour'][:5] )
print(df['weekday'][:5] )
```
![提取日期、小时、星期几](images\prepare_python\date_hour_weekday.png)

```python
print('时间范围：',df['date'].min(),'至',df['date'].max())
```
![时间范围](images\prepare_python\时间范围.png)  
`根据输出结果可知:`   
`行为类型分布严重不均衡：pv占比89.7%，buy仅2% `   
`时间戳存在异常值：最小值1902-05-07，最大值2037-04-09（无效时间）`    
`用户ID范围：1~1,018,011（约100万用户） ` `

### 数据缩小
* 一共有1亿+条数据，考虑到电脑性能不足，计算资源有限，无法处理大量数据，所以抽取部分数据进行分析
* 抽样后的数据保存为sampled_data.csv
```python
import pandas as pd
import numpy as np

chunksize = 100000
p = 0.03   # 抽样比例

reader = pd.read_csv("data/UserBehavior.csv", chunksize=chunksize)

first_chunk = True

for chunk in reader:
    sampled_chunk = chunk[np.random.rand(len(chunk)) < p]

    sampled_chunk.to_csv(
        "data/sampled_data.csv",
        mode="a",
        header=first_chunk,
        index=False
    )

    first_chunk = False
```



### 数据清洗
* 处理缺失值、重复值
* 时间戳标准化（转换为datetime格式）
* 异常值处理（过滤超出时间范围的记录）
* 清洗后剩余数据866,611条

```python
import pandas as pd

# 1.读取数据
file_path='data\\sample_300w.csv'
df=pd.read_csv(file_path)

#2.数据基本信息   
print('\n查看数据结构')
df.info()
print('\n查看前五行数据')
df.head()
```
![查看数据结构和前五行数据](images\prepare_python\head_info.png)

```python
# 3.处理缺失值
print('\n缺失值统计')
print(df.isnull().sum())
df=df.dropna()
```
![缺失值统计](images\prepare_python\handle_isnull_sum.png)

```python
# 4.处理重复值
duplicate_count = df.duplicated().sum()
print('\n重复值数量：',duplicate_count)
df=df.drop_duplicates()
```
![重复值数量](images\prepare_python\duplicate_count.png)

```python
# 5.处理时间字段
if 'timestamp' in df.columns:
    df["timestamp"] = pd.to_datetime(df['timestamp'],unit="s")
    start_date=pd.Timestamp('2017-11-25')
    end_date=pd.Timestamp('2017-12-03')
    df=df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

# 6.重置索引
df = df.reset_index(drop=True)

# 7.保存数据
output_path = 'data\\cleaned_sample_date.csv'
df.to_csv (output_path,index=False)
print("\n清洗完成，最终数据规模:", df.shape)
print("数据已保存到:", output_path)
```
### 数据存储
* 在数据库管理工具中导入清洗后的数据 （这里用的navicat）  
打开Navicat，点击左上角连接MySQL，点击下一步，出现：
![连接mysql](images\anlasis_sql\连接mysql.png)
* 这里连接名称随便写，密码这里是之前下载MySQL的，我的是root，然后单击“测试连接”下方出现的图标，如果是绿色就可以了就点右下角“确定”。  
* 打开连接，右键新建数据库，按下图操作，当然名字你随便起一个就好：
![新建数据库](images\anlasis_sql\新建连接数据库.png)
然后双击数据库，点击右边有一行‘导入向导’，选择CSV文件，除了下图的这一步修改字段名称行为0，第一个数据行为1，都直接跳过就行，这里注意，最后一个数据行这个选项需要根据电脑水平量力而行，最后只用了100w条：
![导入向导](images\anlasis_sql\导入向导.png)
![字段](images\anlasis_sql\字段.png)
导入后为：
![导入后](images\anlasis_sql\导入后.png)

## 阶段2：数据统计与分析
### 用户获取
* PV：PageView，即页面浏览量，用户每次刷新一个网页被计算一次  
* UV：Unique Visitor，即独立访客，指在某一定时间内访问某网站的唯一用户数量  
* PV/UV：浏览深度，即平均每个独立访客浏览的页面数量  
我们根据日期分组统计每天的情况，然后保存到表PUV中。
```sql
CREATE TABLE PUV(
  dates date,
  PV int,
  UV int,
  PV_UV decimal(10,2)
)

INSERT INTO PUV
SELECT 
    DATE(TIMESTAMP) as dates,
    COUNT(*) as PV,
    COUNT(DISTINCT user_id) as UV,
    ROUND(COUNT(*)/COUNT(DISTINCT user_id),2) as PV_UV
FROM 
    cleaned_sample_date
Where 
    behavior_type= 'pv'
GROUP BY 
    dates 
```
运行结果如下：
![用户获取](images\anlasis_sql\用户获取.png)
### 用户留存
由于这份数据只有八天，这里只计算次日留存率和三日留存率
* 次日留存率：某一天新增用户中，在次日仍然登录或使用产品的用户占比
* 三日留存率：某一天新增用户中，注册后第三天仍然活跃的用户比例。
次日留存率部分：
```sql
CREATE TABLE df_retention1(
    dates DATETIME ,
    retention_1 FLOAT
);

INSERT INTO df_retention1
SELECT 
    ub1.dates,
    ROUND(COUNT(ub2.user_id)/COUNT(ub1.user_id
    ),2) as retention_1
FROM(
  SELECT
      DISTINCT user_id,
      DATE(timestamp) as dates
      FROM cleaned_sample_date
)as ub1
LEFT JOIN
(
  SELECT
  DISTINCT user_id,
      DATE(timestamp) as dates
      FROM cleaned_sample_date
)as ub2
on ub1.user_id = ub2.user_id
AND ub2.dates=DATE_ADD(ub1.dates,interval 1 day)
GROUP BY ub1.dates;    
```
因为随机抽样后12月3日的用户很少，所以12月2日的留存率为0，结果仅供参考。  
![次日留存率](images\anlasis_sql\次日留存率.png)  
三日留存率部分：  
```sql
CREATE TABLE df_retention3(
    dates DATETIME ,
    retention_3 FLOAT
);

INSERT INTO df_retention3
SELECT 
    ub1.dates,
    ROUND(COUNT(ub2.user_id)/COUNT(ub1.user_id
    ),2) as retention_3
FROM(
  SELECT
      DISTINCT user_id,
      DATE(timestamp) as dates
      FROM cleaned_sample_date
)as ub1
LEFT JOIN
(
  SELECT
  DISTINCT user_id,
      DATE(timestamp) as dates
      FROM cleaned_sample_date
)as ub2
on ub1.user_id = ub2.user_id
AND ub2.dates=DATE_ADD(ub1.dates,interval 3 day)
GROUP BY ub1.dates;  
```
![三日留存率](images\anlasis_sql\三日留存率.png)
### 用户行为
这里统计每天每时用户群体的各种行为，因此以dates和hours分组
```sql
CREATE TABLE df_timeseries(
    dates date,
    hours int(9),
    PV int(9),
    CART int(9),
    FAV int(9),
    BUY int(9)
);

INSERT INTO df_timeseries
SELECT 
    date(timestamps) as dates,
    hour(timestamps) as hours,
    COUNT(IF(behavior_type= 'pv',1,NULL)) as PV,
    COUNT(IF(behavior_type= 'cart',1,NULL)) as CART,    
    COUNT(IF(behavior_type= 'fav',1,NULL)) as FAV, 
    COUNT(IF(behavior_type= 'buy',1,NULL)) as BUY
FROM cleaned_sample_date
GROUP BY 
    date(timestamps),
    hour(timestamps) 
ORDER BY 
    date(timestamps),
    hour(timestamps) ;
```
![用户行为](images\anlasis_sql\用户行为.png)  

### 用户转化率
先统计每个用户对每个商品都进行了哪些行为，然后以视图形式保存  
```sql
CREATE VIEW user_behavior_total
AS
SELECT
    user_id,
    item_id,
    COUNT(IF(behavior_type='pv',1,NULL)) as PV,
    COUNT(IF(behavior_type='fav',1,NULL)) as FAV,
    COUNT(IF(behavior_type='cart',1,NULL)) as CART,
    COUNT(IF(behavior_type='buy',1,NULL)) as BUY
FROM
    cleaned_sample_date
GROUP BY
    user_id,
    item_id;
```
运行结果如下：  
![](images\anlasis_sql\用户对每个商品的行为.png)
接下来对用户行为归一标准化，有过这种行为标记为1，没有这种行为标记为0
```sql
CREATE VIEW user_behavior_total_standard 
AS
SELECT
    user_id,
    item_id,
    IF(PV > 0,1,0) AS ifpv,
    IF(FAV > 0,1,0) AS iffav,
    IF(CART > 0,1,0) AS ifcart,
    IF(BUY > 0,1,0) AS ifbuy
FROM
    user_behavior_total
GROUP BY
    user_id,
    item_id;
```
然后用concat函数把用户对商品的所有行为合并起来，也就是把原来四列合并成一列，形成一个用户行为路径：
```sql
CREATE VIEW user_path
AS
SELECT
    user_id,
    item_id,
    CONCAT(ifpv,iffav,ifcart,ifbuy) AS path
FROM
    user_behavior_total_standard;
```
运行结果如下，path是1000代表某用户对某商品有过浏览行为：  
![行为标准化](images\anlasis_sql\行为标准化.png)  
有些用户path是0001，代表用户直接购买商品，这可能是用户在统计时间之前有浏览过商品或者把商品加入购物车所以没记录到。

这里用正则表达式进行筛选，然后按照path分组拥挤每种path的数量，为了方便查看，用case when语句加入了注释列
```sql
CREATE VIEW user_path_num 
AS
SELECT
  path,
  case
    WHEN path=1101 THEN '浏览-收藏-/-购买'
    WHEN path=1011 THEN '浏览-/-加购-购买'
    WHEN path=1111 THEN '浏览-收藏-加购-购买'
    WHEN path=1001 THEN '浏览-/-/-购买'
    WHEN path=1010 THEN '浏览-/-加购-/'
    WHEN path=1100 THEN '浏览-收藏-/-/'
    WHEN path=1110 THEN '浏览-收藏-加购-/'
    ELSE '浏览-/-/-/'
    END AS description,
    count(*) as path_num
FROM
    user_path
WHERE
    path REGEXP '^1'
GROUP BY 
    path;
```
* 进行该部分的数据汇总，最后需要以漏斗图的方式进行分析，即浏览中有多少进行了收藏和加入购物车，收藏和加入购物车中又有多少人购买了商品，（后面可以做由多到少的一个漏斗可视化图）
* 所以在这里创建了一个新表，第一列是三种类型，第二列是数量值：
```sql
CREATE TABLE df_buy_path(
    buy_path VARCHAR(55),
    buy_path_num int(9)
);

INSERT INTO df_buy_path
SELECT
    '浏览',
    sum(path_num) as buy_path_num
FROM 
    user_path_num;
    
INSERT INTO df_buy_path 
SELECT
    '浏览后收藏加购',
    SUM(IF(path=1101 OR 
     path=1100 OR
     path=1111 OR
     path=1010 OR
     path=1011 OR
     path=1110,path_num,null)) as buy_path_num
FROM
    user_path_num;
     
INSERT INTO  df_buy_path
SELECT
    '浏览后收藏加购后购买',
    sum(if (path=1101 OR 
      path=1011 OR
      path=1111,path_num,NULL)) as buy_path_num
FROM
    user_path_num;

```
结果如图所示：  
![浏览路径，后续漏斗图准备](images\anlasis_sql\浏览路径.png)  
这里会有偏差，因为除了这些路径还有浏览后直接购买，所以浏览里还需去掉浏览后直接购买量。除此之外还有在统计时段内进行浏览，在统计时段后收藏加入购物车又或是购买。

### 用户定位
用户定位这里采用了常见的RFM分析方法
* R（最近一次消费的时间间隔）
* F（一定时间内的消费频率）
* M（一定时间内的消费金额）给用户分为8类（2的3次方）  
思路如下图所示（图来自猴子的数据分析思维这本书）：
![RFM分析方法](images\anlasis_sql\RFM用户定位.png)  
* 由于数据集没有消费金额，这里只采用前面两个维度（RF）  
* 首先我们计算R值和F值（以视图的形式存储），由于RFM要求用户是消费过的，我们这里筛选出behavior_type为buy的用户，然后按照用户分组：
```sql
-- 计算R
CREATE VIEW R AS
SELECT
    user_id,
    max(date(timestamps)) as 'last_buy_date'
FROM
    cleaned_sample_date
WHERE
    behavior_type='buy'
GROUP BY 
    user_id;
    
-- 计算F
CREATE VIEW F AS
SELECT
    user_id,
    count(user_id) as 'buy_times'
FROM 
    cleaned_sample_date
WHERE
    behavior_type='buy'
GROUP BY 
    user_id;
```
为了方便同时对两组数据进行处理，这里将他们(视图R,F)合并成一个新表
```python
CREATE TABLE df_rfm_model(
    user_id int(9),
    recency date,
    frequency int(9)
);    

INSERT INTO df_rfm_model
SELECT
    user_id,
    last_buy_date,
    buy_times
FROM
    F
JOIN
    R using(user_id);
```
合并结果如图所示：
![rfm结果图](images\anlasis_sql\rfm结果图.png)

下一步需要对每个用户的recency和frequency这两个变量进行打分量化，比如购买次数越多得分越高，这样才能给用户进行分组，这里使用case when语句：
```sql
-- 量化R
ALTER TABLE df_rfm_model ADD r_score INT(9);

UPDATE df_rfm_model
SET r_score =
  CASE
    WHEN recency='2017-12-03' then 100
    WHEN recency='2017-12-02' or recency='2017-12-01' then 80
    WHEN recency='2017-11-30' or recency='2017-11-29' then 60
    WHEN recency='2017-11-28' or recency='2017-11-27' then 40
    ELSE 20
  END;

-- 量化F
ALTER TABLE df_rfm_model ADD f_score INT(9);

UPDATE df_rfm_model 
SET f_score=
  CASE 
    WHEN frequency=10 THEN 100
    WHEN frequency BETWEEN 8 and 9 then 90
    when frequency BETWEEN 6 and 7 then 70
    when frequency BETWEEN 4 and 5 then 50
    when frequency BETWEEN 2 and 3 then 30
    ELSE 10
  END;
```
运行结果：  
![RF分量化](images\anlasis_sql\RF量化.png)

* 猴子的数据分析思维的图，那里面各个坐标轴其实就是整体的平均值，比整体平均值高了就在坐标轴右侧（或上侧等等）
* 所以我们给上一步的表加上两列，希望能表示r_score和f_score的平均值，然后对每一行作比较即可。
* 希望每一行都能显示平均值，所以这里用了窗口函数（不然的话其他行会被聚合掉）
```sql
CREATE TABLE df_rfm_model_avg(
    user_id int(9),
    recency date,
    r_score int(9),
    avg_r FLOAT,
    frequency int(9),
    f_score int(9),
    avg_f FLOAT
    
)

INSERT INTO df_rfm_model_avg  
SELECT
    a.user_id,
    recency,
    r_score,
    a.avg_r,
    frequency,
    f_score,
    a.avg_f
FROM
  (SELECT
      user_id,
      avg(r_score) over() as avg_r,
      avg(f_score) over() as avg_f
  FROM
      df_rfm_model
      ) as a
JOIN 
    df_rfm_model USING(user_id);
```
运行结果如下：
![RF平均值](images\anlasis_sql\rf平均值.png)

avg_f是11分，说明用户的购买频率集中在较低水平，这里可以设置评分标准更细一点，这样会把用户分的更细，avg_f分或许会高点。
* 最后从两个维度把用户分成四类，使用case when语句划分
```sql
CREATE TABLE df_rfm_result(
    user_class VARCHAR(5),
    user_class_num INT(9)
);

INSERT INTO df_rfm_result
SELECT  
    user_class,
    count(*) as user_class_num
FROM
    (SELECT
        *,
      CASE
        WHEN(f_score >= avg_f AND r_score >= avg_r) THEN '价值用户'
        WHEN(f_score >= avg_f AND r_score < avg_r) THEN '保持用户'
        WHEN(f_score < avg_f AND r_score >= avg_r) THEN '发展用户'
        ELSE '挽留用户'
      END AS user_class
    FROM
        df_rfm_model_avg
    ) as g
GROUP BY user_class;    
```
![rf四类用户数量](images\anlasis_sql\rf四类用户数量.png)

### 商品热度
* 这里以浏览类为度量统计了热销top10的商品和品类
```sql
-- 热门品类
CREATE TABLE df_popular_category(
    category_id int,
    category_pv int
);

INSERT INTO df_popular_category
SELECT
    category_id,
    count(if(behavior_type = 'pv',1,null)) as category_pv
FROM
    cleaned_sample_date
WHERE 
    behavior_type='pv' 
GROUP BY 
    category_id
ORDER BY 
    count(if(behavior_type = 'pv',1,null)) DESC
LIMIT 10;
```
![热门品类](images\anlasis_sql\热门品类.png)

```sql
-- 热门商品
CREATE TABLE df_popular_item(
    item_id INT,
    item_pv INT
);

INSERT INTO df_popular_item
SELECT
    item_id,
    count(if(behavior_type='pv',1,null)) as item_pv
FROM
    cleaned_sample_date
GROUP BY 
    item_id
ORDER BY
    count(if(behavior_type='pv',1,null)) DESC
LIMIT 10;
```
结果如图所示：  
![热门商品](images\anlasis_sql\热门商品.png)

### 商品特征
* 这一部分和前面的RFM模型其实很像，也是根据几个指标的平均值进行分类，在商品这里是对品类分类，分完类我们就知道哪些类别商品可能需求量特别大，个人认为可以通过商品的pv、fav、cart、buy四个量相加来进行总量排序代表需求量，那么研发部门的推荐算法可以优先展示。
* 笔者从原表中根据category_id进行分组，统计每个类别是商品被浏览、收藏、加入购物车、购买的次数。
```sql
CREATE TABLE df_product_attributes(
    category_id INT,
    PV INT,
    FAV INT,
    CART INT,
    BUY INT,
    totalsum INT
);

     
INSERT df_product_attributes
SELECT 
    category_id,
    COUNT(if(behavior_type='pv',1,NULL)) as PV,
    COUNT(if(behavior_type='fav',1,NULL)) as FAV,
    COUNT(if(behavior_type='cart',1,NULL)) as CART,
    COUNT(if(behavior_type='buy',1,NULL)) as BUY,
    count(*) AS totalsum
FROM 
    cleaned_sample_date
GROUP BY 
    category_id 
ORDER BY 
    totalsum desc;
```
结果如图所示：  
![商品特征](images\anlasis_sql\商品特征.png)
热门的商品总量可以有上万，冷门商品最低只有1。  
## 阶段3：数据可视化与分析
### 用户获取分析
这里用之前的puv表，把PV和UV制成双轴图。  
* 整体趋势分析   
数据集从2017年11月25日开始，这天是周六，统计周期的前两天和后两天都是周末，PV值和UV值都要比周中高一点，整体趋势要比较稳定。
* 异常波动点   
PV和UV峰值是12月2日，比其他周末要高出很多，推测可能是12月商家提前为‘双十二’活动进行预热。PV值一直有一定波动，而UV值波动不大，可能是同一批用户反复访问，即有很大一部分是老客户，一方面商家做活动偏向于这部分客户，另一方面商家需要做出新变化来吸引新用户。

*  用户粘性分析  
PV/UV记为浏览深度用来判断用户粘性，一方面周末浏览深度的增加一定程度上表明商家的广告、优惠活动成功吸引到了用户，使得用户多次点击查看，另一方面理论上不排除商家活动页面设计更加复杂，用户们需要花费更多时间查看的可能
![用户流量分析](images\visualization\用户流量分析.png)
### 用户留存分析
这个数据源需要把之前的df_retention_1和df_retention_3用dates连接起来
* 留存率在12月之前一直在30%左右，说明用户首次体验有一定吸引力以及用户粘性也比较稳定，进入12月份后明显上升，显著说明商家的活动成功吸引到了用户，为我们排除了之前页面设计更加复杂的推测。

* 本图中留存率迅速衰减的原因是由于数据集跨度不够导致的，但我们可以设想继续分析’双十二‘活动附近几天的用户留存率变化在一定程度上评估某一次活动的效果，进而商家就能知道用户喜欢什么，进而调整自己的策划方案，实现用户的长期价值。
![用户留存率分析](images\visualization\用户留存率分析.png)
### 用户行为分析
这里需要把日期和小时设置一个分层结构，PV绘制堆积图，其余绘制折线图，然后设置双轴
* 从波峰波谷来看，流量高峰出现在中午12-14点，低峰在18-19点，同时购买量、收藏量、加购量呈现出明显的正相关，这也符合一般认知，商家需要做的是在高峰期保证平台正常运行以免给用户带来不好的体验感。

* 收藏量比加购量的要多一半，商家可以针对购物车出一些优惠活动，促进购买量。

* 商家可以针对流量高峰采取行动，例如先是在高峰期采取更多推送，用户不但不反感还更有可能点击查看，然后采用限时限量优惠，比如每天某一个点限量发放满减优惠券以此实现收益扩大化。
![用户单日行为分析](images\visualization\用户单日行为分析.png)
### 用户转化率分析
这里用了漏斗模型，数据集是之前的df_buy_path表。
这里因为路径统计只有浏览后收藏加购后再购买，所以到最后购买量只有几十个，购买率几乎为0，这里从中间两个阶段入手。
* 从浏览到收藏加购来看，大多数用户在浏览后并没有进行收藏或加入购物车。需要分析用户在浏览时遇到的问题，例如产品展示不吸引人，或是促销信息不明确等等，商家需要考虑的是推出限时优惠或打折促销，激励用户尽快收藏和加入购物车，从之前的分析来看收藏加购和购买呈现出明显的正相关性。

* 从收藏加购到购买的转化率仅为0%，表明用户在收藏或加入购物车后，因价格、支付流程、物流问题等等因素未完成购买，建议商家分阶段进行A/B测试，尝试不同的页面设计、促销策略和用户体验优化，找到最佳方案。
![用户转化率分析](images\visualization\用户转化率分析.png)
### 用户定位分析
这里用RFM模型对用户进行分类以圆环图展示，数据集是之前的df_rfm_result表。
* 这里发展用户和挽留用户占比明显高于其他两类用户，代表购买频率没有高于平均
值，商家需要针对这些用户采取不同措施促进这两类用户购买。

* 商家的用户定位分析而言，针对不同的用户群体，他们可以指定不同的策略，比如针对价值用户要继续维持之前的服务不能松懈，针对发展用户，可以采取第二次购买优惠的手段来提高他们的购买频率，针对保持用户，属于一段时间没来的忠实用户，可以采取给这类用户单独发放优惠券的形式来吸引他们，最后对于挽留用户，应尽可能地采取措施挽回，整体上实现精细化运营。
![用户定位分析](images\visualization\用户定位分析.png)
### 商品热度分析
* 商品如果热门的话，平台就给它放在主页推荐位显眼的地方更容易吸引顾客，热门品类也是如此，在购物活动来临时可以加大力度促销这些商品及品类，这里找了TOP10。
![商品热度分析](images\visualization\热门商品和种类.png)
### 商品特征分析
这里的特征分析是用矩阵分析，根据两个指标的平均值对商品的品类进行区分，注意我这里用了筛选器只考虑点击量大于15000的部分，因为当可视化之后会发现在均值交汇处数据非常密集，说明整个数据集受极小值影响很大，因此将他们去除：

* 第一象限（高点击、高购买）  
此类品类表现出高曝光与高转化的双重特征，可判定为高频刚需类商品（如日化家清类产品）。用户对这类产品具有稳定且持续的潜在需求，品类内部商品丰富度高、选择余地大，因此呈现出广泛的用户触达与较高的转化水平。
 
* 第二象限（低点击、高购买）  
该类品类体现出低认知参与度但高决策确定性的特点，用户转化链路极短，意味着用户对品牌 / 品类具有高度认知与强锁定意向。结合高购买率特征，可推断该品类市场集中度较高、替代成本较高，用户倾向于 “即搜即买”，典型场景为强品牌心智、差异化较小的快消品。

* 第四象限（高点击、低购买）  
此类品类呈现高关注度但低转化率，具有典型的低刚性、高弹性属性（例如珠宝类、高端 3C 类产品）。用户在购买前会进行多轮信息搜集与比价，属于高参与度、长决策周期的消费行为，点击量高但最终转化受价格、情绪、场景等因素影响较大。

* 第三象限（低点击、低购买）  
此类品类表现为低触达、低转化，符合低需求频次、高可替代性的特征。用户对该类商品不存在强依赖，线下即时渠道即可满足需求，线上搜索意愿与购买意愿均偏低。这类商品通常属于低决策成本、强日常替代性品类，用户路径短且无需额外信息搜集。
![商品特征分析](images\visualization\商品特征分析.png)
### 仪表盘搭建
![仪表盘](images\visualization\仪表板.png)
## 总结
在数据集统计期间：

1.PV上升的同时UV维持在22K左右，平台流量以老用户为主，吸引新用户能力不足；

2.’双十二‘预热活动有效提升了用户的短期留存率将近4个百分点；

3.在中午12-14点流量高峰期间建议发放限时限量优惠，扩大收益；

4.从浏览到收藏加购再到购买的转化率分别为0.28%和0%，建议通过A/B测试调整策略来提高转化率；

5.对于不同的用户和商品应该实现精细化运营（详见上文）；