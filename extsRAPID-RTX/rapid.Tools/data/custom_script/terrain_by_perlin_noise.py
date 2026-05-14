import omni.kit.commands
from pxr import Usd, UsdGeom, Gf, Vt
import math
import random
import omni.usd


class PerlinNoise:
    def __init__(self, seed=None):
        if seed:
            random.seed(seed)
        self.p = list(range(256))
        random.shuffle(self.p)
        self.p = self.p * 2

    def noise(self, x, y):
        X = int(math.floor(x)) & 255
        Y = int(math.floor(y)) & 255
        x -= math.floor(x)
        y -= math.floor(y)
        u = self.fade(x)
        v = self.fade(y)
        aa = self.p[self.p[X] + Y]
        ab = self.p[self.p[X] + Y + 1]
        ba = self.p[self.p[X + 1] + Y]
        bb = self.p[self.p[X + 1] + Y + 1]
        return self.lerp(v, self.lerp(u, self.grad(aa, x, y), self.grad(ba, x - 1, y)),
                            self.lerp(u, self.grad(ab, x, y - 1), self.grad(bb, x - 1, y - 1)))

    def fade(self, t): return t * t * t * (t * (t * 6 - 15) + 10)
    def lerp(self, t, a, b): return a + t * (b - a)
    def grad(self, hash, x, y):
        h = hash & 3
        if h == 0: return x + y
        elif h == 1: return -x + y
        elif h == 2: return x - y
        else: return -x - y

# --- 地形生成逻辑 ---
def create_perlin_terrain(path="/World/Perlin_Terrain", size=20.0, res=100, scale=4.0, height_mult=2.5):
    """
    scale: 噪声缩放系数，值越大，起伏越频繁（山头越多）
    height_mult: 高度倍率，值越大，山越高
    """
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(path):
        omni.kit.commands.execute('DeletePrims', paths=[path])

    pn = PerlinNoise(seed=42)
    points = []
    face_vertex_counts = []
    face_vertex_indices = []

    step = size / (res - 1)

    # 1. 生成顶点
    for i in range(res):
        for j in range(res):
            x = i * step - size / 2
            y = j * step - size / 2

            # 使用柏林噪声计算 Z 轴 (高度)
            # 叠加两层噪声（分形）可以让地形更真实
            noise_val = pn.noise(i/res * scale, j/res * scale) * 1.0
            noise_val += pn.noise(i/res * scale * 4, j/res * scale * 4) * 0.2 # 细节毛刺

            z = noise_val * height_mult
            points.append(Gf.Vec3f(x, y, z))

    # 2. 生成面索引 (四边形)
    for i in range(res - 1):
        for j in range(res - 1):
            idx0 = i * res + j
            idx1 = i * res + (j + 1)
            idx2 = (i + 1) * res + (j + 1)
            idx3 = (i + 1) * res + j
            face_vertex_indices.extend([idx0, idx1, idx2, idx3])
            face_vertex_counts.append(4)

    # 3. 创建 Mesh
    omni.kit.commands.execute('CreatePrim',
        prim_type='Mesh',
        prim_path=path,
        attributes={
            'points': Vt.Vec3fArray(points),
            'faceVertexIndices': Vt.IntArray(face_vertex_indices),
            'faceVertexCounts': Vt.IntArray(face_vertex_counts),
        }
    )

    # 4. 美化处理
    prim = stage.GetPrimAtPath(path)
    UsdGeom.Mesh(prim)

# 运行脚本
# scale=3.0 (平缓丘陵), height_mult=3.0 (高山)
create_perlin_terrain(size=50.0, res=30, scale=4.5, height_mult=4.0)