import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import math

# 控制是否画图
show_animation = True


# 此类将三维点云数据的指定高度范围转化为2D栅格地图数据
class UAVPlanning:
    def __init__(self) -> None:
        # 点云数据存储位置
        PCDpath = r"C:\Users\ZZZ\Desktop\林上低密度点云无深度.txt"
        # 起始点位置，单轴索引=(现位置-最小位置)/分辨率 ， 分辨率即为每米划分为几个索引
        self.sx = -20.0  # [m]
        self.sy = -20.0  # [m]
        self.gx = 20.0  # [m]
        self.gy = 20.0  # [m]
        self.GridResolution = 0.5  # [m]正方形网格的边长，即分辨率
        robot_radius = 0.5  # [m]机器人半径，用于做边缘膨胀
        # 栅格地图坐标位置范围，建立围墙等
        self.MapRangeX = [-25, 25]
        self.MapRangeY = [-25, 25]
        # 这两个方法完成点云数据转到2D加权栅格地图
        self.PCD(PCDpath)
        ViewObstacleX, ViewObstacleY = self.GridMap(self.GridResolution, robot_radius, self.MapRangeX, self.MapRangeY)
        self.motion = self.get_motion_model()
        # 路径规划
        '''设定多个起止点, 拐弯扫描'''
        '''Start2GoalList = []
        Start2GoalList.append([-20, -20, 20, -20])
        Start2GoalList.append([20, -20, 20, 0])
        Start2GoalList.append([20, 0, -20, 0])
        Start2GoalList.append([-20, 0, -20, 20])
        Start2GoalList.append([-20, 20, 20, 20])
        PathX, PathY = self.planning(Start2GoalList)
        if show_animation:  # pragma: no cover
            plt.plot(ViewObstacleX, ViewObstacleY, ".k")
            plt.plot(self.sx, self.sy, "og")
            plt.plot(self.gx, self.gy, "xb")
            for i in range(len(PathX)):
                plt.plot(PathX[i], PathY[i])
            plt.grid(True)
            plt.axis("equal")
            plt.pause(0.01)
            plt.show()'''

    # 点云滤波得到2D离散点数据
    def PCD(self, PCDpath):
        # 通过对点云的直通滤波和半径滤波得到指定高度范围的点云信息
        pcd = o3d.io.read_point_cloud(PCDpath, format='xyz')  # 这里最好使用绝对路径
        points = np.asarray(pcd.points)
        ind = np.where((points[:, 2] >= 1) & (points[:, 2] <= 1.5))[0]
        z_cloud = pcd.select_by_index(ind)
        sor_pcd, ind = z_cloud.remove_radius_outlier(3, 1)

        # 从滤波结果中提取xy平面值，并过滤掉栅格图范围外的点
        sor_pcd = np.asarray(sor_pcd.points)
        self.x_point = sor_pcd[:, 0]
        self.y_point = sor_pcd[:, 1]

    # 构建加权栅格地图
    def GridMap(self, GridResolution, robot_radius, MapRangeX, MapRangeY):
        # 设定障碍点的位置，构建障碍点图
        ObstacleX, ObstacleY = [], []
        for i in range(min(MapRangeX), max(MapRangeX)):
            ObstacleX.append(i)
            ObstacleY.append(-25.0)
        for i in range(min(MapRangeY), max(MapRangeY)):
            ObstacleX.append(25.0)
            ObstacleY.append(i)
        for i in range(min(MapRangeX), max(MapRangeX)):
            ObstacleX.append(i)
            ObstacleY.append(25.0)
        for i in range(min(MapRangeY), max(MapRangeY)):
            ObstacleX.append(-25.0)
            ObstacleY.append(i)
        ObstacleX.extend(self.x_point)
        ObstacleY.extend(self.y_point)

        # 得到栅格地图有多少行多少列,即最大索引数
        self.x_Index = round((max(MapRangeX)-min(MapRangeX))/GridResolution)
        self.y_Index = round((max(MapRangeY)-min(MapRangeY))/GridResolution)
        # 初始化栅格地图，默认权值为false
        self.ObstacleMap = [[False for _ in range(self.x_Index)]
                            for _ in range(self.y_Index)]

        # 膨胀地图，即比较地图上的每个点和障碍点间的距离，如果小于机器人半径，说明机器人不能通过（true）.
        # 这时self.ObstacleMap就成为了一张加权图
        ViewObstacleX, ViewObstacleY = [], []
        for ix in range(self.x_Index):
            xPositon = ix*GridResolution + min(MapRangeX)  # 计算每个栅格的位置，索引到点的位置
            for iy in range(self.y_Index):
                yPosition = iy*GridResolution + min(MapRangeY)
                for iox, ioy in zip(ObstacleX, ObstacleY):
                    # 只要栅格点xPositon、yPosition与任意一个iox、ioy点的距离小于半径，即加权为true
                    d = math.hypot(iox - xPositon, ioy - yPosition)
                    if d <= robot_radius:
                        # 加上这两行生成加权栅格图(膨胀图)，方便之后查看检验
                        ViewObstacleX.append(xPositon)
                        ViewObstacleY.append(yPosition)
                        self.ObstacleMap[ix][iy] = True
                        break
        return ViewObstacleX, ViewObstacleY

    # dijkstra路径规划算法
    def planning(self, Start2GoalList):
        '''dijkstra path search 基于dijkstra的路径搜索方法'''
        AllPathX, AllPathY = [], []
        for i in Start2GoalList:
            print('running')
            StartX, StartY, GoalX, GoalY = i[0], i[1], i[2], i[3]
            # 初始化节点和节点set，
            StartNode = self.Node(self.Position2Index(StartX, min(self.MapRangeX)),
                                  self.Position2Index(StartY, min(self.MapRangeY)), 0.0, -1)
            GoalNode = self.Node(self.Position2Index(GoalX, min(self.MapRangeX)),
                                 self.Position2Index(GoalY, min(self.MapRangeY)), 0.0, -1)
            OpenSet, ClosedSet = dict(), dict()
            OpenSet[self.Index2Number(StartNode)] = StartNode

            # 最核心的dijkstra算法
            while True:
                # print('running')
                '''min函数以值中的cost项为条件, 筛选最小的键，得到代价最小的节点'''
                minCostNumber = min(OpenSet, key=lambda i: OpenSet[i].cost)
                CurrentNode = OpenSet[minCostNumber]

                # 判断是否为终点
                if CurrentNode.x == GoalNode.x and CurrentNode.y == GoalNode.y:
                    print('find goal')
                    GoalNode.ParentNumber = CurrentNode.ParentNumber
                    GoalNode.cost = CurrentNode.cost
                    break

                # 在openset中删除当前节点，在closedset中加入当前节点
                del OpenSet[minCostNumber]
                ClosedSet[minCostNumber] = CurrentNode

                # 基于运动模型的搜索临近网格,即搜索现在节点的所有临近节点，合格的加入到openset
                for MoveX, MoveY, Cost in self.motion:
                    NextNode = self.Node(CurrentNode.x + MoveX,
                                         CurrentNode.y + MoveY,
                                         CurrentNode.cost + Cost, minCostNumber)
                    NextNodeNumber = self.Index2Number(NextNode)
                    # 下个节点是否在closedset中
                    if NextNodeNumber in ClosedSet:
                        continue
                    # 下个节点是否为地图内的节点
                    if not self.VerifyNode(NextNode):
                        continue
                    # 下个节点不在openset中就加入进去
                    if NextNodeNumber not in OpenSet:
                        OpenSet[NextNodeNumber] = NextNode
                    else:
                        # 下个节点在openset中，并且下个节点的cost较小，替换openset中现有的节点
                        if OpenSet[NextNodeNumber].cost >= NextNode.cost:
                            OpenSet[NextNodeNumber] = NextNode

            # 得到最短路径上每个点的索引
            PathX, PathY = self.FinalPath(GoalNode, ClosedSet)
            AllPathX.append(PathX)
            AllPathY.append(PathY)
        return AllPathX, AllPathY

    class Node:
        def __init__(self, x, y, cost, ParentNumber):
            self.x = x   # index of grid 栅格的索引
            self.y = y
            self.cost = cost  # g(n) 代价
            self.ParentNumber = ParentNumber  # Number of previous Node 上一个节点的索引

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.ParentNumber)

    @staticmethod
    def get_motion_model():
        # 横移，竖移，代价
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)],]
        return motion

    def Index2Position(self, index, minp):
        pos = index * self.GridResolution + minp
        return pos

    def Position2Index(self, position, minp):
        return round((position - minp) / self.GridResolution)

    def Index2Number(self, node):
        '''由索引转到第几个栅格， 例索引(2,2)代表第12个栅格的位置,这不同于真实世界的位置'''
        return node.y * self.x_Index + node.x

    def VerifyNode(self, node):
        '''在搜索临近网格时，判断某个临近网格是否在区域内或者障碍点'''
        px = self.Index2Position(node.x, min(self.MapRangeX))
        py = self.Index2Position(node.y, min(self.MapRangeY))

        # 判断是否在区域内,+-1是为了去掉栅格地图墙的边缘点
        if px < min(self.MapRangeX)+1:
            return False
        if py < min(self.MapRangeY)+1:
            return False
        if px > max(self.MapRangeX)-1:
            return False
        if py > max(self.MapRangeY)-1:
            return False

        # 判断是否时障碍点
        if self.ObstacleMap[node.x][node.y]:
            return False
        return True

    def FinalPath(self, GoalNode, ClosedSet):
        # 将节点的索引转化为位置，方便画图
        PathX = [self.Index2Position(GoalNode.x, min(self.MapRangeX))]
        PathY = [self.Index2Position(GoalNode.y, min(self.MapRangeY))]
        # 最后一个节点的上一个节点
        ParentIndex = GoalNode.ParentNumber
        while ParentIndex != -1:
            node = ClosedSet[ParentIndex]
            PathX.append(self.Index2Position(node.x, min(self.MapRangeX)))
            PathY.append(self.Index2Position(node.y, min(self.MapRangeY)))
            ParentIndex = node.ParentNumber
        return PathX, PathY


# UAVPlanning()
