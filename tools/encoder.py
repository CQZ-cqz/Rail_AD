import socket
import struct
import time

# ---------------------用户设置区域---------------------
TCP_IP = "192.168.1.200"
TCP_PORT = 4001

RESOLUTION = 1024           # 10bit
TRAVEL_PER_REV = 250.0      # mm per revolution

SLAVE_ID = 0x01             # 默认站号 1
REG_ADDR = 0x0000           # 单圈寄存器
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

    try:
        while True:
            sock.send(READ_CMD)
            data = sock.recv(32)
            pos = parse_position(data)
            if pos is None:
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

            print(f"单圈: {pos:4d} | 多圈: {multi_turn_count:+4d} | 总位移: {total_dist_mm:10.3f} mm")
            time.sleep(0.02)  # 50Hz更新

    except KeyboardInterrupt:
        print("\n用户终止")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
