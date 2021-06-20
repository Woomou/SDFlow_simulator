from math import ceil
import matplotlib.pyplot as plt
font = {"family":"SimHei","weight":"bold","size":16}
plt.rc("font",**font)
plt.rc("axes",unicode_minus=False)
import numpy as np
from numpy import random as nr
import sys
import time
import _thread
from random import randint
from random import normalvariate
from tqdm import *
'''
__________________________________________________
sys.setrecursionlimit ( x ) -> 函数递归最高 X 次
SYSTEM_ID -> 区别组织体唯一的ID
___________________________________________________
'''
sys.setrecursionlimit(2000)
thread = _thread.start_new_thread
SYSTEM_id = 1
root_path = "C:\\Users\\Lenovo8\\Desktop\\sim_py\\graph\\"
'''
______________________________________________________________
pair( A , B ) 生成一个pair对象
pair1 == pair2 比较两个pair对象
print(pair) 控制台输出pair对象的信息
______________________________________________________________
'''
class pair():
    x = None
    y = None
    def __init__(self,x,y) -> None:
        self.x = x
        self.y = y
    def __eq__(self, another) -> bool:
        return self.x == another.x and self.y == another.y
    def __hash__(self) -> int:
        return hash(hash(self.x),hash(self.y))
    def __repr__(self) -> str:
        return str(self.x)+str(self.y)
    def __str__(self) -> str:
        return "X: "+str(self.x)+" and "+"Y: "+str(self.y)
'''
___________________________________________________________________
people() 生成一个符合正态分布的初始量人群
people[index] 返回给定人群上的索引数目
len(people) 人群的初始量的总和是多少？
___________________________________________________________________
'''
class people():
    peoples:np.ndarray = None
    peoples_sum:float = None
    scale:int = None
    base:int = None
    def __init__(self,scale,sum_num):
        self.scale = scale
        self.peoples = np.abs(nr.normal(0,1,scale))
        self.peoples_sum = np.sum(self.peoples)
        proportion = sum_num / self.peoples_sum
        for j in range(scale):
            self.peoples[j] *= proportion
        self.peoples_sum = sum_num
        self.peoples.sort()
        self.base = 0
    def __getitem__(self,index) -> float:
        return self.peoples[ index - self.base ]
    def __len__(self) -> float:
        return self.peoples_sum
    ##新的经济量进行更新
    def update(self,new_amount:float) -> None:
        difference = new_amount - self.peoples_sum
        factor:float = difference / float(self.scale)
        for j in range(self.scale):
            self.peoples[j] += factor
        self.peoples_sum = new_amount
        return
'''
______________________________________________________________
organ() 生成一个pair对象
organA & organB 求A与B的共同 子组织体
organA | organB 求A与B的共同 父组织体
print(organ) 控制台输出organ对象的信息
organA == organB ? 比较两个organ对象是否是同一个？
len(organ) 求出organ的组织层序位置
organA in organB ? organ对象A 是否为 organ对象B 的一个child？
id(organ) 输出系统为 organ对象所编的内存码
organ() 求 organ对象上的部分量
organ.get_child() 随机输出 organ对象的某个 child
organ.get_parent() 随机输出 organ对象的某个 parent
organ.add_relation(another_organ,0/1) 对 organ对象的 child(1)添加another或者 parent(0)添加another
organ.remove_relation(another_organ,0/1) 对 organ对象的 child(1)移除another或者 parent(0)移除another
organ.purify_relation() 对organ对象的所有子组织体去重复！
______________________________________________________________
'''
class organ():
    pass
class organ():
    bind:people = None
    sysid = None
    level = None
    child = set()
    parent = set()
    def __init__(self):
        global SYSTEM_id
        self.sysid = SYSTEM_id
        SYSTEM_id += 1
        self.child = set()
        self.parent = set()
        self.level = 0
        self.bind = 0
    def __and__(self,another) -> bool:
        return self.child & another.child
    def __or__(self,another) -> bool:
        return self.parent & another.parent
    def __repr__(self) -> str:
        return "Sys_id:"+str(self.sysid)+" Organ"
    def __str__(self) -> str:
        return "SYS_id:"+str(self.sysid)
    def __eq__(self,another) -> bool:
        return self.sysid == another.sysid
    def __hash__(self) -> int:
        return hash(self.sysid)
    def __len__(self) -> int:
        return len(self.child)
    def __contains__(self,another) -> bool:
        for c in self.child:
            if(another == c):
                return True
        return False
    def __id__(self) -> int:
        return self.sysid
    def __call__(self) -> float:
        if(len(self) == 0):
            return self.bind[self.sysid-1]
        ret = 0
        for k in self.child:
            ret += k()/len(k.parent)
        return ret
    def get_child(self):
        if(len(self.child) > 0):
            ret = self.child.pop()
            self.child.add(ret)
            return ret
        else:
            return 0
    def add_relation(self,another,flag) -> None:
        if(another is self):
            return
        if(flag):##flag -> 1 对子组织体
            self.child.add(another)
            another.parent.add(self)
        else:##flag -> 0 对父组织体
            self.parent.add(another)
            another.child.add(self)
        return
    def remove_relation(self,another,flag) -> None:
        if(flag):##flag -> 1 对子组织体
            try:
                self.child.remove(another)
                another.parent.remove(self)
            except KeyError:
                raise("The organ haven\'t such a child.")
        else:##flag -> 0 对父组织体
            try:
                self.parent.remove(another)
                another.child.remove(self)
            except KeyError:
                raise("The organ haven\'t such a parent.")
        return
    def purify_relation(self) -> None:
        get_child = lambda organ : organ.get_child()
        endpoint = set()
        path = [self]
        while(True):
            if(self in endpoint):
                break
            next = get_child(path[-1])
            if(type(next) is int):
                endpoint.add(path[-1])
                path.pop()
                continue
            if(next in endpoint):
                if(len(path[-1].child - endpoint) <= 0):
                    endpoint.add(path[-1])
                    path.pop()
                continue
            if(next in path):
                path[-1].remove_relation(next,1)
                path.pop()
                continue
            path.append(next)
        self.level = len(endpoint)
        return
'''
______________________________________________________________
lattice(num) 生成一个有 num 个 organ对象的 栅层
lattice[index] 返回指定索引位置的 organ对象
len(lattice) 求栅层内有多少个 organ对象？
for organ in lattice 迭代输出栅层内的 organ对象
organ in lattice ? 测试organ对象是否在栅层中
lattice.add( organ )添加某个特定 organ对象
lattice.delete( id )删除某个 organ 对象
______________________________________________________________
'''
class lattice():
    pass
class lattice():
    panel:dict = None
    size:int = None
    index:int = None
    bind_partion = None
    def __init__(self,organ_num:int,bind_partion) -> None:
        self.panel = dict()
        self.size = 0
        self.index = 1
        while(organ_num > 0):
            self.size += 1
            self.panel["%d"%self.size] = organ()
            organ_num -= 1
        self.bind_partion = bind_partion
    def __getitem__(self,id:int) -> organ:
        if "%d"%id not in self.panel:
            raise IndexError
        return self.panel["%d"%id]
    def __len__(self) -> int:
        return self.size
    def __iter__(self) -> lattice:
        return self
    def __next__(self) -> organ:
        self.index += 1
        if(self.index > self.size):
            self.index = 1
            raise StopIteration
        return self[self.index]
    def __contains__(self) -> bool:
        for i in range(1,len(self)+1):
            if(organ == self[i]):
                return True
        return False
    def __repr__(self) -> print:
        return str(self.panel)
    def __str__(self) -> str:
        return str(self.panel)
    def add(self,organ:organ) -> None:
        self.size += 1
        self.panel["%d"%self.size] = organ
        return
    def delete(self,id:int) -> None:
        if "%d"%id not in self.panel:
            raise("lattice index out of range.")
        del self.panel["%d"%id]
        self.size -= 1
    def evolute(self,direct) -> None:
        key = [int(i) for i in self.panel.keys()]
        for j in key:
            implement( self[j], direct, self.bind_partion )()
        return
'''
___________________________________________________________________
partion() 初始化一个partion对象
partion[index] 返回指定索引位置的栅层
len(partion) 求有多少个栅层？
for lattice in partion 逐个访问划分体内的栅层
partion.init_lattice(organ_num) 根据给定 organ对象的数目初始化一个栅格
partion.init_link(id) 根据所给ID对对应的栅层创建组织体随机连接并去重
___________________________________________________________________
'''
class partion():
    overlap:dict = None
    height:int = None
    index:int = None
    init_times:int = None
    scale:int = None
    scale_range_param:int = None
    threshold:list = None
    def __init__(self):
        self.overlap = dict()
        self.height = 0
        self.index = 0
        self.init_times = 0
        self.scale = 0
        self.scale_range_param = 0
        self.threshold = 0
    def __getitem__(self,order:int) -> lattice:
        if("%d"%order not in self.overlap):
            raise IndexError
        return self.overlap["%d"%order]
    def __len__(self) -> int:
        return len(self.overlap)
    def __iter__(self):
        return self
    def __next__(self) -> lattice:
        if(self.index == self.height):
            raise StopIteration
        self.index += 1
        return self[self.index]
    def __contains__(self,id:int) -> bool:
        for key in self.overlap:
            if("%d"%id == key):
                return True
        return False
    def add_lattice(self,level:int):
        self.height += 1
        self.overlap["%d"%level] = lattice(0,self)
    def del_lattice(self,order:str):
        self.height -= 1
        del self.overlap[order]
    def init_lattice(self,organ_num:int):
        self.height += 1
        self.overlap["%d"%self.height] = lattice(organ_num,self)
    ##必须保证其充分平缓
    def init_link(self,order = 1):
        pair_handle = lambda maximum : maximum - abs(ceil(normalvariate(1,maximum/3)))
        init_lattice = self[order]
        for i in range(self.init_times):
            random_pair = pair_handle(2*self.scale_range_param)
            random_pair = random_pair if random_pair > 0 else 0
            random_pair = pair(2*self.scale_range_param - random_pair,random_pair)
            ##平均更少的执行次数
            for j in range(random_pair.x):
                organ_range = set([init_lattice[randint(1,len(init_lattice))] for k in range(random_pair.y)])
                if(len(organ_range) <= 1):
                    continue
                separate_organ = organ_range.pop()
                for org in organ_range:
                    separate_organ.add_relation(org,randint(0,1))
        for org in init_lattice:
            org.purify_relation()
        return
    def init_update(self,order = 1):
        init_lattice = self[order]
        for org in range(1,len(init_lattice)+1):
            lev = init_lattice[org].level
            if(lev <= 1):
                continue
            if(lev not in self):
                self.add_lattice(lev)
            self[lev].add(init_lattice[org])
            init_lattice.delete(org)
        return
    def init_partion(self,scale:int,link_times:int,scale_range_param:int,people:people,threshold:list):
        self.scale = scale
        self.init_times = link_times
        self.scale_range_param = scale_range_param
        self.threshold = threshold
        self.init_bind(people)
        self.init_lattice(self.scale)
        self.bind_with_people(people)
        self.init_link()
        self.init_update()
        ##完成初始化和更新！
        return
    def init_bind(self,people:people):
        global SYSTEM_id
        ##用基数区别开不同的人群
        people.base = SYSTEM_id
        return
    def bind_with_people(self,people:people,order = 1):
        init_lattice = self[order]
        for org in init_lattice.panel:
            init_lattice.panel[org].bind = people
        return
        ##输入范围参数和数据对
    def search(self, range_pair:pair):
        cooperated = set()
        range_param = self.scale_range_param
        for id in range(range_pair.x,range_pair.y+1):
            if(id not in self):
                continue
            lat = self[id].panel
            for org in lat:
                ##博弈范围有交集
                if((lat[org].level - range_param) < range_pair.y or (lat[org].level + range_param) > range_pair.x):
                    cooperated.add(lat[org])
        return cooperated
    def evolute(self,direct):
        for lat in self.overlap:
            self.overlap[lat].evolute(direct)
            print("%s号栅层已完成更新！"%lat)
        ##重新调整栅层
        lat_id = [i for i in self.overlap.keys()]
        for lat in lat_id:
            latt:lattice = self.overlap[lat]
            id_table = [int(i) for i in latt.panel.keys()]
            for org in id_table:
                lev = latt[org].level
                if(lev <= 1):
                    continue
                if(lev not in self):
                    self.add_lattice(lev)
                self[lev].add(latt[org])
                latt.delete(org)
        ##删除空栅层
        for lat in lat_id:
            if(len(self.overlap[lat]) == 0):
                self.del_lattice(lat)
        return
'''
___________________________________________________________________
implement(organ, direct, flag) 生成基于 organ对象的执行策略对象
implement() 计算最优策略并执行
implement.exclusion() 计算最优化的被排除的某个子（父）组织体并计算效用比较系数
implement.expand() 计算最优化的被新添的某个子（父）组织体并计算效用比较系数
___________________________________________________________________
'''
class implement():
    ##  direct:1 父对子 direct:0 子对父
    direct:int = None
    handler:organ = None
    range_param:int = None
    partioner:partion = None
    def __init__(self,handler:organ,direct:int,partioner:partion) -> None:
        self.handler = handler
        self.direct = direct
        self.range_param = partioner.scale_range_param
        self.partioner = partioner
    ##计算该策略的决策
    ##采用了双阈值法来确定决策
    def __call__(self) -> int:
        contain = self.handler.parent if self.direct == 0 else self.handler.child
        diff = self.partioner.threshold[self.direct] - len(contain)
        if(diff <= 0):##达到阈值情况
            self.handler.remove_relation(self.exclusion(),self.direct)
        elif(len(contain) <= 1):##组织体过少的情况
            self.handler.add_relation(self.expand(),self.direct)
            self.handler.purify_relation()
        else:##随机决策
            if(randint(1,self.partioner.threshold[self.direct]-2) > diff):
                self.handler.remove_relation(self.exclusion(),self.direct)
            else:
                self.handler.add_relation(self.expand(),self.direct)
                self.handler.purify_relation()
        return 0
    def exclusion(self) -> organ:
        contain = self.handler.parent if self.direct == 0 else self.handler.child
        if(len(contain) <= 1):
            return 0
        utility = dict()
        for handlee in contain:
            part_x = 0
            self.handler.remove_relation(handlee,self.direct)
            for org in contain:
                part_x += org()
            self.handler.add_relation(handlee,self.direct)
            utility[handlee] = part_x
        max_u = max(utility,key=utility.get)
        return max_u
    def expand(self) -> organ:
        range_param = self.range_param
        contain = self.handler.parent if self.direct == 0 else self.handler.child
        search_range = pair(self.handler.level - range_param, self.handler.level + range_param)
        search_space = self.partioner.search(search_range)
        search_table = {key() : key for key in search_space}
        while(True):
            if(len(search_table) <= 1):
                for last in search_table:
                    return search_table[last]
                break
            optimal:organ = search_table[max(search_table)]
            optimal_max:dict = {key() : key for key in self.partioner.search( pair(optimal.level - range_param, optimal.level + range_param) )}
            if(len(optimal_max) > 0):
                if( optimal_max[max(optimal_max)] == self.handler ):
                    return optimal
            del search_table[max(search_table)]
        return

'''
___________________________________________________________________
watch(list,int) 建立一个观测量
watch.deriviate() 对观测量求增长率
watch.deriv_speed() 对观测量求二阶增长率
___________________________________________________________________
'''

class watch():
    pass
class watch():
    time_length:int = None
    data:np.ndarray = None
    deriviator:watch = None
    deriv_speedor:watch = None
    def __init__(self,data_table:list,time_length:int) -> None:
        self.data = np.array(data_table)
        self.time_length = time_length
        if(self.time_length != len(self.data)):
            raise IndexError
        self.deriviator = 0
    ##对观测量求增长率
    def deriviate(self) -> None:
        deriv = []
        for i in range(self.time_length-1):
            deriv.append( (self.data[i+1] - self.data[i]) / self.data[i] )
        deriv.append(deriv[-1])##尾部值连续化
        self.deriviator = watch(deriv,len(deriv))
        return
    def deriv_speed(self) -> None:
        if(self.deriviator == None):
            raise FileNotFoundError
        speed = []
        for i in range(self.time_length-1):
            speed.append( self.deriviator.data[i+1] / self.deriviator.data[i] )
        speed.append(speed[-1])##尾部值连续化
        self.deriv_speedor = watch(speed,len(speed))
        return

'''
___________________________________________________________________
SDFlow() 创建一个对流簇，整合基本的分析对象
SDFlow.update() 向前演化一步，并更新基本数据
___________________________________________________________________
'''
class SDFlow():
    ##规模参数
    ##默认实验组 1000，200，5
    scale:int = 50
    link_times:int = 10
    scale_range:int = 5
    ##供需量
    supply:partion = None
    demand:partion = None
    supply_people:people = None
    demand_people:people = None
    supply_quantum:float = 1010.0
    demand_quantum:float = 990.0
    ##演化变量
    t:int = 0
    delta1:float = lambda none : randint(1,10) * 0.3
    delta2:float = lambda none : randint(1,5) * 0.2
    scale_search_param:int = 3
    threshold:list = [5,5]##左为最大父组织体数，右为最大子组织体数
    ##观测量的时间序列
    supply_observation:list = None
    demand_observation:list = None
    def __init__(self) -> None:
        start = time.time()
        sq = self.supply_quantum
        dq = self.demand_quantum
        self.supply_people = people(self.scale,sq)
        self.supply_observation = [sq]
        self.demand_people = people(self.scale,dq)
        self.demand_observation = [dq]
        print("初始化人群共用时间："+str(time.time() - start)+" s.")
        start = time.time()
        self.supply = partion()
        self.demand = partion()
        self.supply.init_partion(self.scale,self.link_times,self.scale_range,self.supply_people,self.threshold)
        self.demand.init_partion(self.scale,self.link_times,self.scale_range,self.demand_people,self.threshold)
        print("初始化划分体共用时间："+str(time.time() - start)+" s.")
    def update(self) -> None:
        self.t += 1
        ##先更新供需数量和时间序列
        supply_t = self.supply_quantum
        demand_t = self.demand_quantum
        supply_t += (self.demand_quantum - self.supply_quantum) * self.delta1()
        demand_t += (self.supply_quantum - self.demand_quantum) * self.delta2()
        self.supply_observation.append(supply_t)
        self.demand_observation.append(demand_t)
        self.supply_quantum = supply_t
        self.demand_quantum = demand_t
        ##对人群进行通知
        self.supply_people.update(supply_t)
        self.demand_people.update(demand_t)
        ##对划分体进行更新，参数为策略方向，为0则子对父，为1则父对子
        self.supply.evolute(0)
        self.demand.evolute(1)
        ##相应的经济量计算
        return

a = SDFlow()
seq = [list() for i in range(4)]
for time in range(30):
    if(time == 0):
        for lat in a.supply.overlap:
            latt = a.supply.overlap[lat]
            for org in latt.panel:
                seq[0].append( (latt.panel[org])() )
    if(time == 9):
        for lat in a.supply.overlap:
            latt = a.supply.overlap[lat]
            for org in latt.panel:
                seq[1].append( (latt.panel[org])() )
    if(time == 19):
        for lat in a.supply.overlap:
            latt = a.supply.overlap[lat]
            for org in latt.panel:
                seq[2].append( (latt.panel[org])() )
    if(time == 29):
        for lat in a.supply.overlap:
            latt = a.supply.overlap[lat]
            for org in latt.panel:
                seq[3].append( (latt.panel[org])() )
    a.update()
b = [0,9,19,29]
for i in range(4):
    seq[i] = np.array(seq[i])
    seq[i].sort()
    seq[i] = seq[i][::-1]
    plt.plot(np.arange(len(seq[i])),seq[i],ls="-",lw=2,color="Red",label="部分量x")
    plt.xlabel("组织体排名")
    plt.ylabel("部分量 x")
    plt.title("第%d步的部分量 x 分布"%b[i])
    plt.savefig(root_path+"ss%d.png"%i)
    plt.close()
'''
seq1 = list()##供给体平均组织层序
seq2 = list()##供给体熵值 熵增率
seq3 = list()##供需两条线
seq4 = list()##全体组织体的平均工资增长率 平均失业率
seq5 = list()##全体组织体的平均物价增速 平均失业率
for time in range(30):##演化时长30步
    seq = 0.0
    lev = 0.0
    a.update()
    for lat in a.supply.overlap:
        latt = a.supply.overlap[lat]
        for org in latt.panel:
            seq += (latt.panel[org])()
            lev += latt.panel[org].level
    seq1.append(lev/30)
    seq2.append(len(a.supply.overlap))
    seq4.append(seq/30)
    seqq = 0.0
    for lat in a.demand.overlap:
        latt = a.demand.overlap[lat]
        for org in latt.panel:
            seqq += (latt.panel[org])()
    seq5.append(seqq/30)
seq3 = [a.supply_observation,a.demand_observation]
seq3[0].pop(0)
seq3[1].pop(0)
seq1_x = np.arange(30)
seq1 = np.array(seq1)
plt.plot(seq1_x,seq1,ls="-",lw=2,color="Black",label="供给体组织层序平均值")
plt.xlabel("时间步数")
plt.ylabel("供给体平均组织层序")
plt.title("供给体平均组织层序演化")
plt.savefig(root_path+"1.png")
plt.close()
seq2:watch = watch(seq2,30)
plt.plot(seq1_x,seq2.data,ls="-",lw=2,color="Green",label="熵值")
plt.xlabel("时间步数")
plt.ylabel("供给体熵值")
plt.title("供给体熵值演化")
plt.savefig(root_path+"2.png")
plt.close()
seq2.deriviate()
plt.plot(seq1_x,seq2.deriviator.data,ls="-",lw=2,color="Yellow",label="供给体熵增率")
plt.xlabel("时间步数")
plt.ylabel("供给体熵增率")
plt.title("供给体熵增率演化")
plt.savefig(root_path+"3.png")
plt.close()
plt.plot(seq1_x,seq3[0],ls="-",lw=2,color="Red",label="供给量")
plt.plot(seq1_x,seq3[1],ls="-",lw=2,color="Blue",label="需求量")
plt.xlabel("时间步数")
plt.ylabel("宏观经济量")
plt.title("供需宏观量演化比较")
plt.savefig(root_path+"4.png")
plt.close()
seq4:watch = watch(seq4,30)
plt.plot(seq1_x,seq4.data,ls="-",lw=2,color="Yellow",label="供给体平均工资")
plt.xlabel("时间步数")
plt.ylabel("平均供给体工资数量")
plt.title("平均供给体工资数量演化")
plt.savefig(root_path+"5.png")
plt.close()
seq5:watch = watch(seq5,30)
plt.plot(seq1_x,seq5.data,ls="-",lw=2,color="Blue",label="需求体平均消费")
plt.xlabel("时间步数")
plt.ylabel("平均需求体消费数量")
plt.title("平均需求体消费数量演化")
plt.savefig(root_path+"6.png")
plt.close()
seq4.deriviate()
seq5.deriviate()
seq4.deriv_speed()
seq5.deriv_speed()
der:watch = seq4.deriviator
der2:watch = seq4.deriviator
d1:watch = seq4.deriv_speedor
d2:watch = seq5.deriv_speedor
plt.plot(seq1_x,der.data,ls="-",lw=2,color="Green",label="供给体平均工资增长率")
plt.xlabel("时间步数")
plt.ylabel("平均工资增速")
plt.title("平均供给体工资增长率演化")
plt.savefig(root_path+"7.png")
plt.close()
plt.plot(seq1_x,der2.data,ls="-",lw=2,color="Red",label="需求体平均工资增长率")
plt.xlabel("时间步数")
plt.ylabel("平均物价增速")
plt.title("平均需求体物价增速演化")
plt.savefig(root_path+"8.png")
plt.close()
plt.plot(seq1_x,d1.data,ls="-",lw=2,color="Green",label="供给体平均失业增速")
plt.plot(seq1_x,d2.data,ls="-",lw=2,color="Yellow",label="需求体平均失业增速")
plt.xlabel("时间步数")
plt.ylabel("平均失业增速")
plt.title("供给体和需求体平均失业率增速演化比较")
plt.savefig(root_path+"9.png")
plt.close()
'''
