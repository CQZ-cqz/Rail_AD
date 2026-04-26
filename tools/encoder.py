import socket
import struct
import time
import sys

try:
    import msvcrt  # Windows only, 用于非阻塞按键
except Exception:
    msvcrt = None

# ---------------------用户设置区域---------------------
TCP_IP = "169.254.88.200"
TCP_PORT = 4001

RESOLUTION = 1024           # 10bit
TRAVEL_PER_REV = 250.0      # mm per revolution

SLAVE_ID = 0x01             # 默认站号 1
REG_ADDR = 0x0000           # 单圈寄存器

# 硬件清零寄存器（如果你的设备支持写寄存器设零，请在这里填写寄存器地址；否则留 None）
HW_ZERO_REG = None  # 例如 0x0100
# -----------------------------------------------------


# CRC16 Modbus-RTU
def crc16_modbus(data):
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x01:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc.to_bytes(2, "little")  # <低位在前>


def build_read_cmd():
    frame = bytes([
        SLAVE_ID,
        0x03,  # 读寄存器
        REG_ADDR >> 8, REG_ADDR & 0xFF,
        0x00, 0x01  # 读取 1 寄存器（2字节）
    ])
    return frame + crc16_modbus(frame)


def build_write_cmd(reg_addr, value):
    """构造 Modbus 单寄存器写入（功能码 0x06）的帧。
    reg_addr: 16位寄存器地址
    value: 16位要写入的值
    """
    frame = bytes([
        SLAVE_ID,
        0x06,
        (reg_addr >> 8) & 0xFF, reg_addr & 0xFF,
        (value >> 8) & 0xFF, value & 0xFF,
    ])
    return frame + crc16_modbus(frame)


READ_CMD = build_read_cmd()


# TCP连接
def connect_tcp():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2.0)
    s.connect((TCP_IP, TCP_PORT))
    print(f"[连接成功] {TCP_IP}:{TCP_PORT}")
    return s


# 解析单圈值
def parse_position(data):
    # min length: 7 bytes   例：01 03 02 XX XX CRC CRC
    if len(data) < 7:
        return None
    if data[1] != 0x03:
        return None
    pos = (data[3] << 8) | data[4]   # 高位在前
    return pos


def main():
    sock = connect_tcp()
    print("开始读取编码器... Ctrl+C 退出\n")

    last_pos = None
    multi_turn_count = 0   # 虚拟多圈计数（软件累加）

    zero_offset_mm = 0.0  # 软件清零偏移（mm）

    print("按键： z=软件清零, r=恢复(清除)软件清零, w=写入硬件清零寄存器(若配置), q=退出")
    if HW_ZERO_REG is None:
        print("注意：HW_ZERO_REG 未配置，按 w 无效。若要启用硬件清零，请在文件开头设置 HW_ZERO_REG。")

    try:
        while True:
            # 非阻塞按键检查（仅在 Windows 上可用）
            if msvcrt is not None and msvcrt.kbhit():
                ch = msvcrt.getwch()  # 支持宽字符
                if ch.lower() == 'z':
                    # 立刻用当前读数作为零点（软件清零）
                    # 如果还没读取到值，则会在下一次循环设置
                    print("请求：软件清零 (将在下一次有效读数时应用)")
                    pending_software_zero = True
                    # 将在读取到 total_dist_mm 后应用
                elif ch.lower() == 'r':
                    zero_offset_mm = 0.0
                    print("已清除软件零点")
                elif ch.lower() == 'w':
                    if HW_ZERO_REG is None:
                        print("硬件清零寄存器未配置，跳过")
                    else:
                        # 使用当前单圈位置写入硬件寄存器作为零点（设备依赖）
                        try:
                            # 发送读取命令一次以获取当前 pos
                            sock.send(READ_CMD)
                            data = sock.recv(32)
                            pos = parse_position(data)
                            if pos is None:
                                print("读取失败，无法写入硬件清零")
                            else:
                                # 将单圈值写入 HW_ZERO_REG（16位）
                                write_cmd = build_write_cmd(HW_ZERO_REG, pos & 0xFFFF)
                                sock.send(write_cmd)
                                resp = sock.recv(32)
                                # 简单检查回应函数码是否与请求一致
                                if len(resp) >= 2 and resp[1] == 0x06:
                                    print(f"已写入硬件清零寄存器 0x{HW_ZERO_REG:04X} = {pos}")
                                else:
                                    print("写入硬件清零寄存器，设备无响应或返回错误")
                        except Exception as e:
                            print(f"写入硬件清零失败: {e}")
                elif ch.lower() == 'q':
                    raise KeyboardInterrupt

            # 读取编码器
            sock.send(READ_CMD)
            data = sock.recv(32)
            pos = parse_position(data)
            if pos is None:
                time.sleep(0.02)
                continue

            if last_pos is not None:
                diff = pos - last_pos

                # 正转跨界（如 1022 → 3）
                if diff < -RESOLUTION/2:
                    multi_turn_count += 1

                # 反转跨界（如 3 → 1022）
                elif diff > RESOLUTION/2:
                    multi_turn_count -= 1

            last_pos = pos

            # 累计总位移
            total_pos = multi_turn_count * RESOLUTION + pos
            total_dist_mm = total_pos * (TRAVEL_PER_REV / RESOLUTION)

            # 如果有 pending_software_zero 标记，在第一次有效读数时设置零点
            if 'pending_software_zero' in locals() and pending_software_zero:
                zero_offset_mm = total_dist_mm
                pending_software_zero = False
                print(f"软件零点已设置: {zero_offset_mm:10.3f} mm")

            display_mm = total_dist_mm - zero_offset_mm

            print(f"单圈: {pos:4d} | 多圈: {multi_turn_count:+4d} | 总位移: {total_dist_mm:10.3f} mm | 显示(已清零): {display_mm:10.3f} mm")
            time.sleep(0.02)  # 50Hz更新

    except KeyboardInterrupt:
        print("\n用户终止")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
