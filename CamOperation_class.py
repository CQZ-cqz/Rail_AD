# -- coding: utf-8 --
import threading
import time
import sys
import ctypes
import os
import cv2
import numpy as np
from ctypes import *

sys.path.append(os.getenv('MVCAM_COMMON_RUNENV') + "/Samples/Python/MvImport")
from hik_SDK.MvCameraControl_class import *

class CameraOperation():

    def __init__(self, obj_cam, st_device_list, n_connect_num=0, b_open_device=False, b_start_grabbing = False,h_thread_handle=None,\
                b_thread_opened=False, st_frame_info=None, b_save_bmp=False, b_save_jpg=False, buf_save_image=None):

        self.obj_cam = obj_cam  # 相机对象
        self.st_device_list = st_device_list  # 设备列表
        self.n_connect_num = n_connect_num  # 连接编号
        self.b_open_device = b_open_device  # 设备打开状态标志
        self.b_start_grabbing = b_start_grabbing  # 图像采集状态标志
        self.b_thread_opened = b_thread_opened  # 线程开启状态标志
        self.st_frame_info = MV_FRAME_OUT_INFO_EX()  # 帧信息结构体
        self.b_save_bmp = b_save_bmp  # BMP保存标志
        self.b_save_jpg = b_save_jpg  # JPG保存标志
        self.buf_save_image = buf_save_image  # 图像数据缓冲区
        self.buf_save_image_len = 0  # 图像缓冲区长度
        self.h_thread_handle = h_thread_handle  # 线程句柄
        self.buf_lock = threading.Lock()  # 存、取图buffer锁，缓冲区线程锁，防止多线程竞争
        self.exit_flag = 0  # 线程退出标志
        self.frame_count = 0  # 帧计数
        self.lost_frame_count = 0  # 丢帧计数
        self.b_frame_saving = False  # 是否正在抽帧保存
        self.save_interval = 10      # 保存间隔（每多少帧保存一次）
        self.save_frame_count = 0    # 保存帧计数器
        self.saved_count = 0         # 已保存的图片计数
        self.inference_callback = None  # 推理回调函数
        self.inference_interval = 1  # 推理抽帧间隔（相机线程侧）
        self.inference_frame_count = 0
        self.last_frame = None  # 最后一帧图像

    # 将数字转为16进制字符串
    def to_hex_str(self, num):
        chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
        hexStr = ""
        if num < 0:
            num = num + 2**32  # 处理负数
        while num >= 16:
            digit = num % 16
            hexStr = chaDic.get(digit, str(digit)) + hexStr
            num //= 16
        hexStr = chaDic.get(num, str(num)) + hexStr
        return hexStr

    # 打开相机
    def open_device(self):
        if self.b_open_device is False:
            # ch:选择设备并创建句柄 | en:Select device and create handle
            nConnectionNum = int(self.n_connect_num)
            # 获取指定连接编号的设备信息
            stDeviceList = cast(self.st_device_list.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents
            self.obj_cam = MvCamera()  # 创建相机对象
            ret = self.obj_cam.MV_CC_CreateHandle(stDeviceList)  # 创建句柄
            if ret != 0:
                self.obj_cam.MV_CC_DestroyHandle()  # 销毁句柄
                return ret

            ret = self.obj_cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0) # 以独占模式打开设备
            if ret != 0:
                self.b_open_device = False  
                self.b_thread_opened = False
                return ret
            self.b_open_device = True   # 设置设备打开标志
            self.b_thread_opened = False    # 重置线程打开标志

            # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
            if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
                nPacketSize = self.obj_cam.MV_CC_GetOptimalPacketSize()  # 获取最佳数据包大小
                if int(nPacketSize) > 0:
                    ret = self.obj_cam.MV_CC_SetIntValue("GevSCPSPacketSize",nPacketSize)   # 设置数据包大小
                    if ret != 0:
                        print("warning: set packet size fail! ret[0x%x]" % ret)
                else:
                    print("warning: packet size is invalid[%d]" % nPacketSize)

            stBool = c_bool(True)  # 帧率使能bool值设置
            ret = self.obj_cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", stBool)
            # ret = self.obj_cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)  # 帧率使能设置
            if ret != 0:
                print("warning: get acquisition frame rate enable fail! ret[0x%x]" % ret)

            # ch:设置触发模式为off | en:Set trigger mode as off
            ret = self.obj_cam.MV_CC_SetEnumValueByString("TriggerMode", "Off")
            if ret != 0:
                print("warning: set trigger mode off fail! ret[0x%x]" % ret)
            return 0
        return 0
            
    # 开始取图
    def start_grabbing(self, n_index,  win_handle):
        if not self.b_start_grabbing and self.b_open_device:
            ret = self.obj_cam.MV_CC_StartGrabbing()
            if ret != 0:
                self.b_start_grabbing = False
                return ret
            self.b_start_grabbing = True  # 设置采集标志
            print("start grabbing " + str(n_index) + "successfully!")
            try:    # 创建并启动采集线程
                self.exit_flag = threading.Event()  # 线程退出事件
                self.h_thread_handle = threading.Thread(target=CameraOperation.work_thread, args=(self, n_index, win_handle, self.exit_flag))
                self.h_thread_handle.start()
                self.b_thread_opened = True  # 设置线程开启标志
            except TypeError:
                print('error: unable to start thread')
                self.b_start_grabbing = False
            return 0
        return MV_E_CALLORDER

    # 停止取图
    def stop_grabbing(self):
        if (self.b_start_grabbing is True) and (self.b_open_device is True):
            # 退出线程
            if self.b_thread_opened:
                self.exit_flag.set()  # 设置退出标志
                self.h_thread_handle.join()  # 等待线程结束
                self.b_thread_opened = False  # 重置线程标志
            ret = self.obj_cam.MV_CC_StopGrabbing()
            if ret != 0:
                return ret
            self.b_start_grabbing = False  # 重置采集标志
            return 0
        return MV_E_CALLORDER

    # 关闭相机
    def close_device(self):
            
        if self.b_open_device:
            #退出线程
            if self.b_thread_opened:
                self.exit_flag.set()
                self.h_thread_handle.join()
                self.b_thread_opened = False
            if self.b_start_grabbing:   # 如果正在采集，先停止采集
                ret = self.obj_cam.MV_CC_StopGrabbing()
                if ret != 0:
                    return ret
                self.b_start_grabbing = False
            ret = self.obj_cam.MV_CC_CloseDevice()
            if ret != 0:
                return ret
                
        # ch:销毁句柄 | Destroy handle
        self.obj_cam.MV_CC_DestroyHandle()
        self.b_open_device = False
        return 0

    # 设置触发模式
    def set_trigger_mode(self, trigger_mode):
        if True == self.b_open_device:
            if "continuous" == trigger_mode:    # 设置连续采集模式
                ret = self.obj_cam.MV_CC_SetEnumValueByString("TriggerMode","Off")
                if ret != 0:
                    return ret
                return 0
            if "triggermode" == trigger_mode:   # 设置触发模式
                ret = self.obj_cam.MV_CC_SetEnumValueByString("TriggerMode","On")
                if ret != 0:
                    return ret
                return 0

    # 设置触发源
    def set_trigger_source(self, trigger_source):
        if self.b_open_device is True:
            if "software" == trigger_source:
                ret = self.obj_cam.MV_CC_SetEnumValueByString("TriggerSource", "Software")
                if ret != 0:
                    return ret
                return 0
            else:
                ret = self.obj_cam.MV_CC_SetEnumValueByString("TriggerSource", "Line0")
                if ret != 0:
                    return ret
                return 0

    # 软触发一次
    def trigger_once(self):
        if self.b_open_device is True:
            ret = self.obj_cam.MV_CC_SetCommandValue("TriggerSoftware") # 发送软触发命令
            return ret
        
    # 设置相机宽度
    def set_width(self, width_value):
        if self.b_open_device is True:
            ret = self.obj_cam.MV_CC_SetIntValueEx("Width", int(width_value))
            if ret != 0:
                print('set width fail! ret = ' + self.to_hex_str(ret))
            return ret
        return MV_E_CALLORDER

    # 设置相机高度
    def set_height(self, height_value):
        if self.b_open_device is True:
            ret = self.obj_cam.MV_CC_SetIntValueEx("Height", int(height_value))
            if ret != 0:
                print('set height fail! ret = ' + self.to_hex_str(ret))
            return ret
        return MV_E_CALLORDER

    # 设置水平偏移量
    def set_offset_x(self, offset_x_value):
        if self.b_open_device is True:
            ret = self.obj_cam.MV_CC_SetIntValueEx("OffsetX", int(offset_x_value))
            if ret != 0:
                print('set offset x fail! ret = ' + self.to_hex_str(ret))
            return ret
        return MV_E_CALLORDER

    # 设置垂直偏移量
    def set_offset_y(self, offset_y_value):
        if self.b_open_device is True:
            ret = self.obj_cam.MV_CC_SetIntValueEx("OffsetY", int(offset_y_value))
            if ret != 0:
                print('set offset y fail! ret = ' + self.to_hex_str(ret))
            return ret
        return MV_E_CALLORDER

    # 获取宽度参数
    def get_width(self):
        if self.b_open_device is True:
            stIntValue = MVCC_INTVALUE_EX()
            ret = self.obj_cam.MV_CC_GetIntValueEx("Width", stIntValue)
            if ret == 0:
                return stIntValue.nCurValue
            else:
                print('get width fail! ret = ' + self.to_hex_str(ret))
                return -1
        return -1

    # 获取高度参数
    def get_height(self):
        if self.b_open_device is True:
            stIntValue = MVCC_INTVALUE_EX()
            ret = self.obj_cam.MV_CC_GetIntValueEx("Height", stIntValue)
            if ret == 0:
                return stIntValue.nCurValue
            else:
                print('get height fail! ret = ' + self.to_hex_str(ret))
                return -1
        return -1

    # 获取水平偏移量
    def get_offset_x(self):
        if self.b_open_device is True:
            stIntValue = MVCC_INTVALUE_EX()
            ret = self.obj_cam.MV_CC_GetIntValueEx("OffsetX", stIntValue)
            if ret == 0:
                return stIntValue.nCurValue
            else:
                print('get offset x fail! ret = ' + self.to_hex_str(ret))
                return -1
        return -1

    # 获取垂直偏移量
    def get_offset_y(self):
        if self.b_open_device is True:
            stIntValue = MVCC_INTVALUE_EX()
            ret = self.obj_cam.MV_CC_GetIntValueEx("OffsetY", stIntValue)
            if ret == 0:
                return stIntValue.nCurValue
            else:
                print('get offset y fail! ret = ' + self.to_hex_str(ret))
                return -1
        return -1

    # 批量设置ROI参数（宽度、高度、偏移量）
    def set_roi_parameters(self, width, height, offset_x=0, offset_y=0):
        if self.b_open_device is True:
            # 停止采集（如果需要修改这些参数，通常需要先停止采集）
            was_grabbing = self.b_start_grabbing
            if was_grabbing:
                self.stop_grabbing()
            
            # 设置宽高
            ret1 = self.set_width(width)
            ret2 = self.set_height(height)

            # 设置偏移量
            ret3 = self.set_offset_x(offset_x)
            ret4 = self.set_offset_y(offset_y)
            
            # 如果之前正在采集，重新开始采集
            if was_grabbing:
                # 注意：这里需要传入正确的窗口句柄，调用者需要确保提供
                # self.start_grabbing(self.n_connect_num, win_handle)
                pass
                
            # 返回综合结果
            if ret1 == 0 and ret2 == 0 and ret3 == 0 and ret4 == 0:
                return 0
            else:
                return -1
        return MV_E_CALLORDER

    def set_exposure_time(self, str_value):
        if self.b_open_device is True:
            self.obj_cam.MV_CC_SetEnumValue("ExposureAuto", 0)
            time.sleep(0.2)
            ret = self.obj_cam.MV_CC_SetFloatValue("ExposureTime", float(str_value))
            if ret != 0:
                print('show error', 'set exposure time fail! ret = ' + self.to_hex_str(ret))
                return ret
        return 0

    def set_gain(self, str_value):
        if self.b_open_device is True:
            self.obj_cam.MV_CC_SetEnumValue("GainAuto", 0)
            time.sleep(0.2)
            ret = self.obj_cam.MV_CC_SetFloatValue("Gain", float(str_value))
            if ret != 0:
                print('show error', 'set gain fail! ret = ' + self.to_hex_str(ret))
                return ret
        return 0

    def set_frame_rate(self, str_value):
        if self.b_open_device is True:
            ret = self.obj_cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(str_value))
            return ret
        return 0
    
    # 开始抽帧保存
    def start_frame_saving(self, interval=10):
        if self.b_open_device and self.b_start_grabbing:
            self.b_frame_saving = True
            self.save_interval = interval
            self.save_frame_count = 0
            self.saved_count = 0
            return 0
        return -1

    # 停止抽帧保存
    def stop_frame_saving(self):
        self.b_frame_saving = False
        return 0

    # 取图线程函数
    def work_thread(self, n_index, win_handle, exit_flag):
        stOutFrame = MV_FRAME_OUT()  # 输出帧结构体
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))  # 初始化结构体

        # 增加图像缓冲区数量，减少丢帧
        self.obj_cam.MV_CC_SetImageNodeNum(10)

        while not exit_flag.is_set():
            ret = self.obj_cam.MV_CC_GetImageBuffer(stOutFrame, 1000)   # 获取图像缓冲区，超时1000ms
            if 0 == ret:
                # 添加帧同步检查
                self.frame_count += 1

                # 拷贝图像和图像信息
                with self.buf_lock:
                    if self.buf_save_image_len < stOutFrame.stFrameInfo.nFrameLen:
                        if self.buf_save_image is not None:
                            del self.buf_save_image
                            self.buf_save_image = None
                        self.buf_save_image = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
                        self.buf_save_image_len = stOutFrame.stFrameInfo.nFrameLen

                    # 拷贝帧信息和图像数据（使用 memmove，避免 byref(array) 的不确定行为）
                    ctypes.memmove(addressof(self.st_frame_info), addressof(stOutFrame.stFrameInfo), sizeof(MV_FRAME_OUT_INFO_EX))
                    ctypes.memmove(addressof(self.buf_save_image), stOutFrame.pBufAddr, self.st_frame_info.nFrameLen)

                # 使用Display接口显示图像
                stDisplayParam = MV_DISPLAY_FRAME_INFO()  # 显示参数结构体
                memset(byref(stDisplayParam), 0, sizeof(stDisplayParam))  # 初始化
                stDisplayParam.hWnd = int(win_handle)  # 显示窗口句柄
                stDisplayParam.nWidth = stOutFrame.stFrameInfo.nWidth  # 图像宽度
                stDisplayParam.nHeight = stOutFrame.stFrameInfo.nHeight  # 图像高度
                stDisplayParam.enPixelType = stOutFrame.stFrameInfo.enPixelType  # 像素格式
                stDisplayParam.pData = stOutFrame.pBufAddr  # 图像数据指针
                stDisplayParam.nDataLen = stOutFrame.stFrameInfo.nFrameLen  # 数据长度
                display_ret = self.obj_cam.MV_CC_DisplayOneFrame(stDisplayParam)
                if display_ret != 0:
                    print("Camera[" + str(n_index) + "]: display failed, ret = " + self.to_hex_str(display_ret))

                # 抽帧保存逻辑
                if self.b_frame_saving:
                    self.save_frame_count += 1
                    if self.save_frame_count >= self.save_interval:
                        # 保存当前帧
                        ret_save = self.save_bmp_frame_saving()
                        if ret_save == 0:
                            self.saved_count += 1
                            print(f"Camera {n_index} saved frame {self.saved_count}")
                        self.save_frame_count = 0  # 重置计数器

                # 调用推理回调（如果设置了的话）
                if self.inference_callback is not None:
                    self.inference_frame_count += 1
                    interval = max(1, int(self.inference_interval))
                    if self.inference_frame_count % interval == 0:
                        current_frame = self.get_current_frame()
                        if current_frame is not None:
                            try:
                                self.inference_callback(current_frame)
                            except Exception as e:
                                print(f"推理回调错误: {str(e)}")

                # 释放缓存
                self.obj_cam.MV_CC_FreeImageBuffer(stOutFrame)
            else:  # 获取图像失败
                self.lost_frame_count += 1  # 丢帧计数
                if ret != MV_E_NOENOUGH_BUF:  # 忽略缓冲区不足的警告
                    print("Camera[" + str(n_index) + "]:no data, ret = "+self.to_hex_str(ret))
                continue

    # 存BMP图像
    def save_bmp(self):
        if 0 == self.buf_save_image:
            return

        # 获取缓存锁
        self.buf_lock.acquire()

        file_dir = "save_bmp/cam" + str(self.n_connect_num) + "/"
        img_path = str(self.st_frame_info.nFrameNum) + ".bmp"
        os.makedirs(file_dir, exist_ok=True)
        file_path = os.path.join(file_dir, img_path)
        c_file_path = file_path.encode('ascii')

        stSaveParam = MV_SAVE_IMAGE_TO_FILE_PARAM_EX()
        stSaveParam.enPixelType = self.st_frame_info.enPixelType  # ch:相机对应的像素格式 | en:Camera pixel type
        stSaveParam.nWidth = self.st_frame_info.nWidth  # ch:相机对应的宽 | en:Width
        stSaveParam.nHeight = self.st_frame_info.nHeight  # ch:相机对应的高 | en:Height
        stSaveParam.nDataLen = self.st_frame_info.nFrameLen  # 数据长度
        stSaveParam.pData = cast(self.buf_save_image, POINTER(c_ubyte))  # 图像数据指针
        stSaveParam.enImageType = MV_Image_Bmp  # ch:需要保存的图像类型 | en:Image format to save
        stSaveParam.pcImagePath = ctypes.create_string_buffer(c_file_path)  # 文件路径
        stSaveParam.iMethodValue = 1  # 保存方法
        ret = self.obj_cam.MV_CC_SaveImageToFileEx(stSaveParam) # 保存图像到文件

        self.buf_lock.release()

        return ret
    
    # 抽帧保存BMP图像
    def save_bmp_frame_saving(self):
        if 0 == self.buf_save_image:
            return -1

        # 获取缓存锁
        self.buf_lock.acquire()

        # 使用时间戳和计数器生成文件名，避免重复
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # file_path = f"cam{self.n_connect_num}_frame_{self.saved_count:04d}_{timestamp}.bmp"

        file_dir = "save_bmp/cam" + str(self.n_connect_num) + "/"
        img_path = str(self.st_frame_info.nFrameNum) + str(timestamp) + ".png"
        # img_path = str(self.frame_count) + str(timestamp) + ".bmp"
        os.makedirs(file_dir, exist_ok=True)
        file_path = os.path.join(file_dir, img_path)
        c_file_path = file_path.encode('ascii')

        stSaveParam = MV_SAVE_IMAGE_TO_FILE_PARAM_EX()
        stSaveParam.enPixelType = self.st_frame_info.enPixelType
        stSaveParam.nWidth = self.st_frame_info.nWidth
        stSaveParam.nHeight = self.st_frame_info.nHeight
        stSaveParam.nDataLen = self.st_frame_info.nFrameLen
        stSaveParam.pData = cast(self.buf_save_image, POINTER(c_ubyte))
        stSaveParam.enImageType = MV_Image_Png  # 保存为JPG格式
        stSaveParam.pcImagePath = ctypes.create_string_buffer(c_file_path)
        stSaveParam.iMethodValue = 1
        ret = self.obj_cam.MV_CC_SaveImageToFileEx(stSaveParam)

        self.buf_lock.release()

        return ret
    
    # 开始缺陷检测
    def start_defect_detection(self):
        if self.b_open_device and self.b_start_grabbing and not self.b_defect_detection:
            self.b_defect_detection = True
            self.defect_count = 0
            self.detection_exit_flag.clear()
            
            # 启动检测线程
            self.detection_thread = threading.Thread(target=self.defect_detection_thread)
            self.detection_thread.start()
            return 0
        return -1

    # 停止缺陷检测
    def stop_defect_detection(self):
        self.b_defect_detection = False
        if self.detection_thread is not None:
            self.detection_exit_flag.set()
            self.detection_thread.join(timeout=2.0)
            self.detection_thread = None
        return 0

    # 缺陷检测线程
    def defect_detection_thread(self):
        """缺陷检测主线程"""
        print(f"Camera {self.n_connect_num} 开始缺陷检测")
        
        while not self.detection_exit_flag.is_set() and self.b_defect_detection:
            # 获取当前图像数据进行检测
            if self.buf_save_image is not None and self.buf_save_image_len > 0:
                try:
                    # 获取缓存锁
                    self.buf_lock.acquire()
                    
                    # 调用缺陷检测算法
                    has_defect = self.detect_defect_in_frame()
                    
                    if has_defect:
                        # 保存缺陷图片
                        self.save_defect_image()
                        self.defect_count += 1
                        
                    self.buf_lock.release()
                    
                except Exception as e:
                    print(f"Camera {self.n_connect_num} 缺陷检测错误: {str(e)}")
                    self.buf_lock.release()
            
            # 控制检测频率，避免过于频繁
            time.sleep(0.1)  # 每秒检测10次
        
        print(f"Camera {self.n_connect_num} 停止缺陷检测")

    def save_defect_image(self):
        """保存缺陷图片"""
        try:
            import datetime
            
            # 生成带时间戳的文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"defect_cam{self.n_connect_num}_{timestamp}_{self.defect_count:04d}.bmp"
            filepath = os.path.join("defect_images", filename)
            
            # 使用现有的保存功能
            c_file_path = filepath.encode('ascii')
            
            stSaveParam = MV_SAVE_IMAGE_TO_FILE_PARAM_EX()
            stSaveParam.enPixelType = self.st_frame_info.enPixelType
            stSaveParam.nWidth = self.st_frame_info.nWidth
            stSaveParam.nHeight = self.st_frame_info.nHeight
            stSaveParam.nDataLen = self.st_frame_info.nFrameLen
            stSaveParam.pData = cast(self.buf_save_image, POINTER(c_ubyte))
            stSaveParam.enImageType = MV_Image_Bmp
            stSaveParam.pcImagePath = ctypes.create_string_buffer(c_file_path)
            stSaveParam.iMethodValue = 1
            
            ret = self.obj_cam.MV_CC_SaveImageToFileEx(stSaveParam)
            
            if ret == 0:
                print(f"Camera {self.n_connect_num} 保存缺陷图片: {filename}")
            else:
                print(f"Camera {self.n_connect_num} 保存缺陷图片失败! 错误码: {self.to_hex_str(ret)}")
                
            return ret
            
        except Exception as e:
            print(f"保存缺陷图片错误: {str(e)}")
            return -1

    def get_current_frame(self):
        """获取当前帧（用于推理）"""
        if self.buf_save_image is not None and self.buf_save_image_len > 0:
            try:
                with self.buf_lock:
                    pixel_type = self.st_frame_info.enPixelType
                    width = int(self.st_frame_info.nWidth)
                    height = int(self.st_frame_info.nHeight)
                    frame_len = int(self.st_frame_info.nFrameLen)
                    raw = bytes(self.buf_save_image[:frame_len])

                if pixel_type == PixelType_Gvsp_BGR8_Packed:
                    img_np = np.frombuffer(raw, dtype=np.uint8)
                    self.last_frame = img_np.reshape((height, width, 3)).copy()
                    return self.last_frame

                if pixel_type == 17301513:  # PixelType_Gvsp_BayerRG8
                    img_data = np.frombuffer(raw, dtype=np.uint8, count=width * height)
                    img_data = img_data.reshape((height, width))
                    self.last_frame = cv2.cvtColor(img_data, cv2.COLOR_BAYER_RG2BGR)
                    return self.last_frame

                if pixel_type == PixelType_Gvsp_Mono8:
                    img_data = np.frombuffer(raw, dtype=np.uint8, count=width * height)
                    img_data = img_data.reshape((height, width))
                    self.last_frame = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
                    return self.last_frame

                print(f"Unsupported pixel type: {pixel_type}")
                return None
                
            except Exception as e:
                print(f"获取当前帧错误: {str(e)}")
                return None
        return None

    def set_inference_callback(self, callback):
        """设置推理回调函数"""
        self.inference_callback = callback
        self.inference_frame_count = 0
        print(f"Camera {self.n_connect_num} 推理回调已设置")

    def set_inference_interval(self, interval):
        """设置推理抽帧间隔（相机线程侧）。"""
        try:
            iv = int(interval)
        except Exception:
            iv = 1
        self.inference_interval = iv if iv > 0 else 1
        self.inference_frame_count = 0

