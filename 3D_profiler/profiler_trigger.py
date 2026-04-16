from mecheye.shared import *  # 导入Mech-Eye SDK共享模块
from mecheye.profiler import *  # 导入轮廓仪主模块
from mecheye.profiler_utils import *  # 导入轮廓仪工具函数
import cv2  # 导入OpenCV用于图像处理
import numpy as np  # 导入numpy用于数值计算
from time import sleep  # 导入sleep用于延时
from multiprocessing import Lock  # 导入锁用于多进程同步

mutex = Lock()  # 创建全局互斥锁，用于多线程/多进程环境下的数据同步

# 自定义数据采集回调类
class CustomAcquisitionCallback(AcquisitionCallbackBase):
    def __init__(self, width):
        AcquisitionCallbackBase.__init__(self)  # 调用父类构造函数
        self.profile_batch = ProfileBatch(width)    # 创建轮廓数据批次对象

    def run(self, batch):
        mutex.acquire()  # 获取锁，确保线程安全
        if (not batch.get_error_status().is_ok()):
            print("Error occurred during data acquisition.")  # 打印错误信息
            show_error(batch.get_error_status())  # 显示错误信息
        self.profile_batch.append(batch)  # 将采集的数据批次添加到总批次中
        mutex.release()  # 释放锁

# 主类：通过软件和编码器触发轮廓仪数据采集
class TriggerWithSoftwareAndEncoder(object):
    def __init__(self):
        self.profiler = Profiler()  # 创建轮廓仪对象

    def set_timed_exposure(self, exposure_time: int):
        # Set the exposure mode to timed，设置曝光模式为定时模式
        show_error(self.user_set.set_enum_value(
            ExposureMode.name, ExposureMode.Value_Timed))

        # Set the exposure time to {exposure_time} μs，设置曝光时间（微秒）
        show_error(self.user_set.set_int_value(
            ExposureTime.name, exposure_time))

    def set_hdr_exposure(self, exposure_time: int, proportion1: float, proportion2: float, first_threshold: float, second_threshold: float):
        # Set the "Exposure Mode" parameter to "HDR"，设置曝光模式为HDR
        show_error(self.user_set.set_enum_value(
            ExposureMode.name, ExposureMode.Value_HDR))

        # Set the total exposure time to {exposure_time} μs，设置总曝光时间
        show_error(self.user_set.set_int_value(
            ExposureTime.name, exposure_time))

        # Set the proportion of the first exposure phase to {proportion1}%，设置第一阶段曝光比例
        show_error(self.user_set.set_float_value(
            HdrExposureTimeProportion1.name, proportion1))

        # Set the proportion of the first + second exposure phases to {proportion2}% (that is, the
        # second exposure phase occupies {proportion2 - proportion1}%, and the third exposure phase
        # occupies {100 - proportion2}% of the total exposure time)，设置前两阶段累计曝光比例
        show_error(self.user_set.set_float_value(
            HdrExposureTimeProportion2.name, proportion2))

        # Set the first threshold to {first_threshold}. This limits the maximum grayscale value to
        # {first_threshold} after the first exposure phase is completed.设置第一阶段灰度阈值
        show_error(self.user_set.set_float_value(
            HdrFirstThreshold.name, first_threshold))

        # Set the second threshold to {second_threshold}. This limits the maximum grayscale value to
        # {second_threshold} after the second exposure phase is completed.设置第二阶段灰度阈值
        show_error(self.user_set.set_float_value(
            HdrSecondThreshold.name, second_threshold))

    def set_encoder_trigger(self, trigger_direction: int, trigger_signal_counting_mode: int, trigger_interval: int):
        # Set the trigger source to Encoder，设置线扫描触发源为编码器
        show_error(self.user_set.set_enum_value(
            LineScanTriggerSource.name, LineScanTriggerSource.Value_Encoder))
        # Set the encoder trigger direction to {trigger_direction}，设置编码器触发方向
        show_error(self.user_set.set_enum_value(
            EncoderTriggerDirection.name, trigger_direction))
        # Set the encoder signal counting mode to be {trigger_signal_counting_mode}，设置编码器触发信号计数模式
        show_error(self.user_set.set_enum_value(
            EncoderTriggerSignalCountingMode.name, trigger_signal_counting_mode))
        # Set the encoder trigger interval to {trigger_interval}，设置编码器触发间隔
        show_error(self.user_set.set_int_value(
            EncoderTriggerInterval.name, trigger_interval))

    def set_parameters(self):
        self.user_set = self.profiler.current_user_set()    # 获取当前用户设置对象

        # Set the exposure mode to timed
        # Set the exposure time to 100 μs
        self.set_timed_exposure(100)    # 设置定时曝光，曝光时间100微秒

        """
        你也可以使用HDR曝光模式，激光轮廓仪在一个轮廓采集过程中分三个阶段曝光。
        在此模式下，您需要设置总曝光时间、三个曝光阶段的比例以及两个灰度值阈值。
        HDR曝光模式的相关参数设置代码在以下注释中给出。
        """
        # # Set the "Exposure Mode" parameter to "HDR"
        # # Set the total exposure time to 100 μs
        # # Set the proportion of the first exposure phase to 40%
        # # Set the proportion of the first + second exposure phases to 80% (that is, the second
        # # exposure phase occupies 40%, and the third exposure phase occupies 20% of the total
        # # exposure
        # # Set the first threshold to 10. This limits the maximum grayscale value to 10 after the
        # # first exposure phase is completed.
        # # Set the second threshold to 60. This limits the maximum grayscale value to 60 after the
        # # second exposure phase is completed.
        # self.set_hdr_exposure(100, 40, 80, 10, 60)

        # # Enable outlier removal and adjust the outlier removal intensity.
        # # Set the "EnableOutlierRemoval" parameter to true
        # show_error(self.user_set.set_bool_value(EnableOutlierRemoval.name,True))
        # # Set the "OutlierRemovalIntensity" parameter to "VeryLow"，设置离群点去除强度为VeryLow
        # show_error(self.user_set.set_enum_value(OutlierRemovalIntensity.name,OutlierRemovalIntensity.Value_VeryLow))

        # Set the "Data Acquisition Trigger Source" parameter to "Software"，设置数据采集触发源为软件触发
        show_error(self.user_set.set_enum_value(
            DataAcquisitionTriggerSource.name, DataAcquisitionTriggerSource.Value_Software))

        # Set the "Line Scan Trigger Source" parameter to "Encoder"，设置线扫描触发源为编码器
        # Set the (encoder) "Trigger Direction" parameter to "Both"，设置编码器触发方向为双边沿触发
        # Set the (encoder) "Trigger Signal Counting Mode" parameter to "1×"，设置编码器计数模式为1倍
        # Set the (encoder) "Trigger Interval" parameter to 10，设置编码器触发间隔为10
        self.set_encoder_trigger(EncoderTriggerDirection.Value_ChannelALeading,
                                 EncoderTriggerSignalCountingMode.Value_Multiple_1, 10)

        # Set the "Scan Line Count" parameter (the number of lines to be scanned) to 20000，设置扫描线数为20000
        show_error(self.user_set.set_int_value(ScanLineCount.name, 20000))

        # Set the "Laser Power" parameter to 100，设置激光功率为100
        show_error(self.user_set.set_int_value(LaserPower.name, 100))
        # Set the "Analog Gain" parameter to "Gain_2"，设置模拟增益为2倍
        show_error(self.user_set.set_enum_value(
            AnalogGain.name, AnalogGain.Value_Gain_3))
        # Set the "Digital Gain" parameter to 0，设置数字增益为0
        show_error(self.user_set.set_int_value(DigitalGain.name, 0))

        # Set the "Minimum Grayscale Value" parameter to 50，设置最小灰度值为50
        show_error(self.user_set.set_int_value(MinGrayscaleValue.name, 20))
        # Set the "Minimum Laser Line Width" parameter to 2，设置最小激光线宽为2
        show_error(self.user_set.set_int_value(MinLaserLineWidth.name, 2))
        # Set the "Maximum Laser Line Width" parameter to 20，设置最大激光线宽为20
        show_error(self.user_set.set_int_value(MaxLaserLineWidth.name, 20))
        # Set the "Spot Selection" parameter to "Strongest"，设置光点选择为最强点
        show_error(self.user_set.set_enum_value(
            SpotSelection.name, SpotSelection.Value_Strongest))

        # This parameter is only effective for firmware 2.2.1 and below. For firmware 2.3.0 and above,
        # adjustment of this parameter does not take effect.
        # Set the minimum laser line intensity to 10，设置最小光点强度为51（仅对固件2.2.1及以下有效）
        show_error(self.user_set.set_int_value(MinSpotIntensity.name, 51))
        # This parameter is only effective for firmware 2.2.1 and below. For firmware 2.3.0 and above,
        # adjustment of this parameter does not take effect.
        # Set the maximum laser line intensity to 205，设置最大光点强度为205（仅对固件2.2.1及以下有效）
        show_error(self.user_set.set_int_value(MaxSpotIntensity.name, 205))

        """
        Set the "Gap Filling" parameter to 16, which controls the size of the gaps that can be filled
        in the profile. When the number of consecutive data points in a gap in the profile is no
        greater than 16, this gap will be filled.
        设置间隙填充参数为16，控制可以填充的间隙大小
        """
        show_error(self.user_set.set_int_value(GapFilling.name, 16))
        """
        Set the "Filter" parameter to "Mean". The "Mean Filter Window Size" parameter needs to be set
        as well. This parameter controls the window size of mean filter. If the "Filter" parameter is
        set to "Median", the "Median Filter Window Size" parameter needs to be set. This parameter
        controls the window size of median filter.
        设置滤波器为均值滤波，同时需要设置均值滤波窗口大小
        """
        show_error(self.user_set.set_enum_value(
            Filter.name, Filter.Value_Mean))
        # Set the "Mean Filter Window Size" parameter to 2，设置均值滤波窗口大小为2
        show_error(self.user_set.set_enum_value(
            MeanFilterWindowSize.name, MeanFilterWindowSize.Value_WindowSize_2))

        error, self.data_width = self.user_set.get_int_value(
            DataPointsPerProfile.name)  # 获取每个轮廓的数据点数
        show_error(error)

        error, self.capture_line_count = self.user_set.get_int_value(
            ScanLineCount.name)  # 获取扫描线数
        show_error(error)

        error, data_acquisition_trigger_source = self.user_set.get_enum_value(
            DataAcquisitionTriggerSource.name)  # 获取数据采集触发源
        show_error(error)
        self.is_software_trigger = data_acquisition_trigger_source == DataAcquisitionTriggerSource.Value_Software  # 判断是否为软件触发

    def acquire_profile_data(self) -> bool:
        """
        调用start_acquisition()使激光轮廓仪进入采集准备状态，
        然后调用trigger_software()开始数据采集（软件触发）
        """
        print("Start data acquisition.")  # 开始数据采集
        status = self.profiler.start_acquisition()  # 开始采集
        if not status.is_ok():
            show_error(status)
            return False

        if self.is_software_trigger:
            status = self.profiler.trigger_software()  # 软件触发
            if (not status.is_ok()):
                show_error(status)
                return False

        self.profile_batch.clear()  # 清空轮廓批次
        self.profile_batch.reserve(self.capture_line_count)  # 预留空间

        while self.profile_batch.height() < self.capture_line_count:  # 循环直到采集足够数量的轮廓
            # Retrieve the profile data，检索轮廓数据
            batch = ProfileBatch(self.data_width)
            status = self.profiler.retrieve_batch_data(batch)
            if status.is_ok():
                self.profile_batch.append(batch)  # 添加批次数据
                sleep(0.2)  # 延时200ms
            else:
                show_error(status)
                return False

        print("Stop data acquisition.")  # 停止数据采集
        status = self.profiler.stop_acquisition()  # 停止采集
        if not status.is_ok():
            show_error(status)
        return status.is_ok()

    def acquire_profile_data_using_callback(self) -> bool:
        self.profile_batch.clear()  # 清空轮廓批次

        # Set a large CallbackRetrievalTimeout，设置较大的回调检索超时时间
        show_error(self.user_set.set_int_value(
            CallbackRetrievalTimeout.name, 60000))

        self.callback = CustomAcquisitionCallback(self.data_width)  # 创建回调对象

        # Register the callback function，注册回调函数
        status = self.profiler.register_acquisition_callback(
            self.callback)
        if not status.is_ok():
            show_error(status)
            return False

        # Call the start_acquisition to take the laser profiler into the acquisition ready status
        # 调用start_acquisition使激光轮廓仪进入采集准备状态
        print("Start data acquisition")
        status = self.profiler.start_acquisition()
        if not status.is_ok():
            show_error(status)
            return False

        if self.is_software_trigger:
            status = self.profiler.trigger_software()  # 软件触发
            if not status.is_ok():
                show_error(status)
                return False

        while True:
            mutex.acquire()  # 获取锁
            if self.callback.profile_batch.is_empty():  # 检查批次是否为空
                mutex.release()
                sleep(0.5)  # 等待500ms
            else:
                mutex.release()
                break

        print("Stop data acquisition.")  # 停止数据采集
        status = self.profiler.stop_acquisition()
        if not status.is_ok():
            show_error(status)
        self.profile_batch.append(self.callback.profile_batch)  # 添加回调数据
        return True
    
    def continuous_segmented_capture(self, segment_lines=1000, delay_between_segments=2.0, max_segments=10):
        """
        连续采集，分段保存数据
        :param segment_lines: 每个分段采集的线数
        :param delay_between_segments: 分段间延时（秒）
        :param max_segments: 最大分段数
        """
        print("启动连续分段采集模式")
        
        # 设置连续采集模式（ScanLineCount=0表示无限连续）
        show_error(self.user_set.set_int_value(ScanLineCount.name, 0))
        
        # 启动采集
        status = self.profiler.start_acquisition()
        if not status.is_ok():
            show_error(status)
            return False
        
        # 软件触发开始采集
        if self.is_software_trigger:
            status = self.profiler.trigger_software()
            if not status.is_ok():
                show_error(status)
                return False
        
        segment_batch = ProfileBatch(self.data_width)
        segment_count = 0
        
        try:
            while segment_count < max_segments:
                print(f"开始采集第 {segment_count + 1} 个分段，目标线数: {segment_lines}")
                
                # 采集指定线数的数据
                while segment_batch.height() < segment_lines:
                    batch = ProfileBatch(self.data_width)
                    status = self.profiler.retrieve_batch_data(batch)
                    if status.is_ok():
                        segment_batch.append(batch)
                        
                        # 可选：打印编码器信息
                        if batch.height() > 0:
                            first_profile = batch.get_profile(0)
                            last_profile = batch.get_profile(batch.height() - 1)
                            print(f"当前分段进度: {segment_batch.height()}/{segment_lines}, "
                                # f"编码器范围: {first_profile.encoder_value()} - {last_profile.encoder_value()}"
                            )
                        
                        sleep(0.1)  # 减少查询频率
                    else:
                        show_error(status)
                        break
                
                # 保存当前分段数据
                if segment_batch.height() > 0:
                    depth_file = f"depth_segment_{segment_count + 1}.tiff"
                    intensity_file = f"intensity_segment_{segment_count + 1}.png"
                    self.save_depth_and_intensity(depth_file, intensity_file)
                    
                    # 保存点云
                    save_point_cloud(profile_batch=segment_batch, user_set=self.user_set,
                                save_ply=True, save_csv=True, is_organized=True,
                                ply_filename=f"pointcloud_segment_{segment_count + 1}.ply")
                    
                    print(f"第 {segment_count + 1} 个分段保存完成，共 {segment_batch.height()} 条轮廓线")
                    
                    # 清空当前分段，准备下一个分段
                    segment_batch.clear()
                    segment_count += 1
                    
                    # 分段间延时
                    if segment_count < max_segments:
                        print(f"等待 {delay_between_segments} 秒后开始下一个分段...")
                        sleep(delay_between_segments)
        
        except KeyboardInterrupt:
            print("用户中断采集")
        
        finally:
            # 停止采集
            print("停止数据采集")
            status = self.profiler.stop_acquisition()
            if not status.is_ok():
                show_error(status)
        
        return True

    def save_depth_and_intensity(self, depth_file_name, intensity_file_name):
        cv2.imwrite(depth_file_name,
                    self.profile_batch.get_depth_map().data())  # 保存深度图
        cv2.imwrite(intensity_file_name,
                    self.profile_batch.get_intensity_image().data())  # 保存强度图

    def main(self):
        if not find_and_connect(self.profiler):  # 查找并连接设备
            return -1

        if not confirm_capture():  # 确认采集
            return -1

        self.set_parameters()  # 设置参数

        self.profile_batch = ProfileBatch(self.data_width)  # 创建轮廓批次对象

        # Acquire profile data without using callback，不使用回调采集轮廓数据
        if not self.acquire_profile_data():
            return -1

        # # Acquire the profile data using the callback function，使用回调函数采集轮廓数据
        # if not self.acquire_profile_data_using_callback():
            # return -1

        if self.profile_batch.check_flag(ProfileBatch.BatchFlag_Incomplete):
            print("Part of the batch's data is lost, the number of valid profiles is:",
                  self.profile_batch.valid_height())  # 打印有效轮廓数量

        print("Save the depth map and intensity image")  # 保存深度图和强度图
        self.save_depth_and_intensity("depth.tiff", "intensity.png")
        save_point_cloud(profile_batch=self.profile_batch, user_set=self.user_set,
                         save_ply=True, save_csv=True, is_organized=True)  # 保存点云数据

        # # Uncomment the following line to save a virtual device file using the ProfileBatch acquired.
        # # 取消注释以下行以使用获取的ProfileBatch保存虚拟设备文件
        # self.profiler.save_virtual_device_file(self.profile_batch, "test.mraw")

        self.profiler.disconnect()  # 断开连接
        print("Disconnected from the Mech-Eye Profiler successfully")
        return 0
    
    def main_1(self):
        if not find_and_connect(self.profiler):
            return -1

        if not confirm_capture():
            return -1

        self.set_parameters()
        
        # 使用连续分段采集模式
        self.continuous_segmented_capture(
            segment_lines=1000,      # 每个分段1000条轮廓线
            delay_between_segments=5, # 分段间等待5秒
            max_segments=10          # 最多采集10个分段
        )
        
        self.profiler.disconnect()
        print("采集完成并断开连接")
        return 0

if __name__ == '__main__':
    a = TriggerWithSoftwareAndEncoder()  # 创建主类实例
    # a.main()  # 运行主程序
    a.main_1()  # 运行分段采集程序