import os
import numpy as np

"""生成卫星星座的两行轨道元素(TLE)数据。TLE 是描述卫星轨道的标准格式，包含卫星轨道参数，可用于计算卫星在任意时间的位置"""

class TLEGenerator:
    def __init__(self, output_file, planes, satellites_per_plane, inclo, altitude,argpo_init=90,nodeo_phase=None,argpo_phase=True):
        self.output_file = output_file
        self.satellites_per_plane = satellites_per_plane
        self.planes = planes#轨道面总数
        self.inclo = inclo#轨道倾角（度)
        self.altitude = altitude#轨道高度（公里)
        self.mean_motion = self.mean_motion(altitude)#平均运动（每天绕地球的圈数）
        self.ecco = "0002000" # 偏心率（TLE格式，省略小数点，此处为0.0002000）
        self.bstar = "12222-5" # B*阻力项（TLE格式，表征大气阻力影响）
        self.argpo_init=argpo_init# 近地点幅角初始值（度）
        self.nodeo_phase=nodeo_phase# 升交点赤经相位偏移参数
        self.argpo_phase=argpo_phase# 近地点幅角是否按轨道面奇偶调整的开关

    def mean_motion(self,altitude):#计算卫星的平均运动（TLE 中的关键参数，单位：圈 / 天）
        earth_radius = 6371.0
        gravitational_constant = 398600.4418
        semi_major_axis = earth_radius + altitude
        period = 2 * np.pi * np.sqrt((semi_major_axis ** 3) / gravitational_constant)
        period_days = period / (60 * 60 * 24)
        mean_motion = 1 / period_days
        return mean_motion

    def tle_checksum(self, line):#计算 TLE 每行的校验和，确保数据格式正确性
        checksum = 0
        for c in line[:-1]:
            if c.isdigit():
                checksum += int(c)
            if c == '-':
                checksum += 1
        return checksum % 10

    def generate_tles(self):
        with open(self.output_file, "w") as f:
            for i in range(self.planes):
                nodeo = (180 if self.inclo>=75 else 360) / self.planes * i + (0 if not self.nodeo_phase else 360/ self.planes/self.nodeo_phase[1]*self.nodeo_phase[0])
                for j in range(self.satellites_per_plane):
                    # 卫星名称：格式为 "Satellite1_高度_轨道面编号_卫星编号"
                    sat_name = f"Satellite1_{self.altitude}_{i+1}_{j+1}"
                    f.write(sat_name + "\n")
                    """ 生成TLE第一行
                        格式:1 卫星编号U 国际标识符 历元时间 平均运动一阶导数 二阶导数 B*阻力项 校验和
                    """
                    line1 = "1 {0:05d}U 00000A   23121.00000000  .00000000  00000+0  {1} 0  999".format(i * 100 + j + 1, self.bstar)
                    line1 += str(self.tle_checksum(line1 + " ")) + "\n"# 附加校验和
                    f.write(line1)
                    # 平近点角（mo）：卫星在轨道上的位置，按卫星数量平均分配（0~360度）
                    mo = 360.0 / self.satellites_per_plane * j
                    # 近地点幅角（argpo）：根据轨道面奇偶性调整（若argpo_phase为True）
                    argpo = self.argpo_init  if i%2==0 or not self.argpo_phase else self.argpo_init-180/self.satellites_per_plane
                    """ 生成TLE第二行
                        格式:2 卫星编号 倾角 升交点赤经 偏心率 近地点幅角 平近点角 平均运动 校验和
                    """
                    line2 = "2 {0:05d} {1:8.4f} {2:8.4f} {3} {4:8.4f} {5:8.4f} {6:11.8f}  999".format(i * 100 + j + 1, self.inclo, nodeo, self.ecco, argpo, mo, self.mean_motion)
                    line2 += str(self.tle_checksum(line2 + " ")) + "\n"
                    f.write(line2)
        print(f"TLEs written to {os.path.abspath(self.output_file)}")

tle_gen = TLEGenerator("Satellite_Data/60Degree_500_12x24_tles_1.txt", 12, 24, 60,500,argpo_init=90,nodeo_phase=None,argpo_phase=True)
tle_gen.generate_tles()