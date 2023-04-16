class Node:
    def __init__(self, value):
        self.value = value
        self.children = []

count = 0  # 初始化路径计数器
root = Node(0)  # 创建虚拟根节点，使得所有节点都有父节点

def dfs(node, parent, num, visited):
    if values[node] == 1:  # 如果该节点的权值为1，则将其对应的二进制位设置为1
        num = (num << 1) + 1
    else:  # 如果该节点的权值为0，则将其对应的二进制位设置为0
        num = num << 1
    
    if node not in visited and l <= num <= r and parent is not None:  # 判断当前路径代表的二进制数是否在区间范围内
        global count
        count += 1
    
    visited.add(node)  # 将当前节点添加到已访问集合中
    
    for child in tree[node]:  # 遍历当前节点的子节点
        if child != parent:  # 避免重复访问父节点
            dfs(child, node, num, visited)
    
    visited.remove(node)  # 在返回时，将当前节点从已访问集合中移除

# 读入输入数据，并构建树的邻接表表示
n, l, r = map(int, input().strip().split())
values = list(map(int, input().strip()))
tree = [[] for _ in range(n)]
for i in range(n - 1):
    u, v = map(int, input().strip().split())
    tree[u - 1].append(v - 1)
    tree[v - 1].append(u - 1)

# 从每个节点开始深度优先搜索
for i in range(n):
    dfs(i, None, 0, set())

print(count)


