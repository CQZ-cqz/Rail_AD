import serial
import struct
import time

# ========== 配置参数 ==========
SERIAL_PORT = 'COM1'        # 改成你工控机实际的COM口，例如 'COM5'
BAUDRATE = 9600             # 默认9600，如果改过请同步
SLAVE_ID = 1                # 编码器地址（默认1）
RESOLUTION = 1024           # 10bit = 1024 脉冲/圈
TIMEOUT = 1                 # 串口读取超时（秒）

# ========== 辅助函数：计算Modbus CRC16 ==========
def crc16_modbus(data: bytes) -> int:
    """返回CRC16校验值（低字节在前）"""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    # 返回低字节在前的高16位值，以便直接拼接到命令中
    return crc

def build_read_command(slave_id, reg_addr, reg_count):
    """
    构建Modbus 0x03读保持寄存器的命令
    reg_addr: 寄存器起始地址 (如 0x0000)
    reg_count: 读取寄存器个数 (1个32位值即2个寄存器？注意：单圈值占用1个寄存器，即16位)
    """
    cmd = bytearray()
    cmd.append(slave_id)                # 地址
    cmd.append(0x03)                    # 功能码 读保持寄存器
    cmd.append((reg_addr >> 8) & 0xFF)  # 寄存器地址高字节
    cmd.append(reg_addr & 0xFF)         # 寄存器地址低字节
    cmd.append((reg_count >> 8) & 0xFF) # 寄存器数量高字节
    cmd.append(reg_count & 0xFF)        # 寄存器数量低字节
    crc = crc16_modbus(cmd)
    cmd.append(crc & 0xFF)              # CRC低字节
    cmd.append((crc >> 8) & 0xFF)       # CRC高字节
    return bytes(cmd)

def parse_single_turn_value(response, slave_id):
    """
    解析从站返回的数据，返回单圈原始值（int）
    response: 从串口读取的字节串
    返回 (成功标志, 单圈值)
    """
    if len(response) < 5:
        return False, None
    # 检查地址和功能码
    if response[0] != slave_id or response[1] != 0x03:
        return False, None
    byte_count = response[2]   # 数据字节数（应为2）
    if byte_count != 2:
        return False, None
    # 提取数据（高字节在前，大端）
    value = struct.unpack('>H', response[3:5])[0]
    # 可选：验证CRC（这里省略，用于调试可加）
    return True, value

# ========== 主程序 ==========
def main():
    try:
        # 打开串口
        ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUDRATE,
            bytesize=8,
            parity='N',
            stopbits=1,
            timeout=TIMEOUT
        )
        print(f"已打开串口 {SERIAL_PORT}，参数：{BAUDRATE},8,N,1")
    except Exception as e:
        print(f"打开串口失败：{e}")
        return

    # 构建读取单圈值的命令（寄存器地址 0x0000，读取1个寄存器）
    cmd = build_read_command(SLAVE_ID, 0x0000, 1)
    print(f"发送命令: {cmd.hex().upper()}")

    # 连续读取10次，每次间隔0.5秒
    for i in range(10):
        ser.write(cmd)
        resp = ser.read(8)          # 标准回复长度：地址(1)+功能(1)+字节数(1)+数据(2)+CRC(2)=7~8字节，读8够了
        if len(resp) == 0:
            print(f"第{i+1}次: 无响应")
        else:
            success, raw_value = parse_single_turn_value(resp, SLAVE_ID)
            if success and raw_value is not None:
                angle = raw_value * 360.0 / RESOLUTION
                print(f"第{i+1}次: 原始值 = {raw_value:5d}  →  角度 = {angle:.2f}°")
            else:
                print(f"第{i+1}次: 解析失败，原始回复: {resp.hex().upper()}")
        time.sleep(0.5)

    ser.close()
    print("串口已关闭")

if __name__ == "__main__":
    main()