# Go2 真机部署

## 1. 连接 Go2 拓展坞

笔记本有线网卡配置一次静态地址，其中 `enp5s0` 按实际网卡名修改：

```bash
sudo nmcli con add type ethernet ifname enp5s0 con-name go2-static \
  ipv4.addresses 192.168.123.222/24 ipv4.method manual
sudo nmcli con up go2-static
```

连接拓展坞：

```bash
ping -c 3 192.168.123.18
ssh unitree@192.168.123.18
```

在拓展坞确认 Go2 下位机走 `eth0`：

```bash
ip route get 192.168.123.161
ping -c 3 192.168.123.161
```

预期路由包含：

```text
192.168.123.161 dev eth0 src 192.168.123.18
```

## 2. 配置部署环境

真机部署不需要安装 MuJoCo。拓展坞使用 Python 3.8 环境：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate go2ladder_lkx_deploy
```

不要用普通 pip 覆盖 Jetson 的 CUDA PyTorch。检查环境：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import pyrealsense2; import serial; import onnx; print('dependencies ok')"
which trtexec || ls /usr/src/tensorrt/bin/trtexec
```

安装与训练 checkpoint 匹配的 `rsl_rl`，使用 `--no-deps`：

```bash
cd ~/lkx/rsl_rl
python -m pip install -e . --no-deps
python -c "from rsl_rl.modules.student_actor_critic import StudentActorCritic; print('rsl_rl ok')"
```

触觉串口没有权限时执行一次，然后重新登录：

```bash
sudo usermod -aG dialout unitree
```

## 3. 上传代码和模型

以下命令在笔记本执行：

```bash
cd ~/unitree_mujoco
rsync -av deploy/ unitree@192.168.123.18:~/lkx/deploy/
rsync -av ~/rsl_rl/ unitree@192.168.123.18:~/lkx/rsl_rl/
rsync -av /path/to/model.pt unitree@192.168.123.18:~/lkx/deploy/student/
```

在 `deploy_student_config.py` 中把 `checkpoint` 设置为上传的模型，例如：

```python
checkpoint: Optional[Path] = STUDENT_DIR / "model_50000.pt"
```

checkpoint、`rsl_rl` 和部署配置必须来自同一版网络。

## 4. 构建 TensorRT engine

engine 应在实际部署的 Orin 上构建。推荐使用 split FP32 engine：

```bash
cd ~/lkx/deploy
conda activate go2ladder_lkx_deploy
python export_student_tensorrt.py \
  --checkpoint student/model_50000.pt \
  --split-engine \
  --build-engine
```

成功时最后会显示：

```text
[trt-export] split TensorRT verified ...
```

并生成：

```text
student/model_50000_depth_fp32.engine
student/model_50000_core_fp32.engine
```

如果出现 `size mismatch`，说明模型、`rsl_rl` 或配置不匹配；不要用 `strict=False` 跳过。

## 5. 部署

先分别检查相机和触觉，检查后用 `Ctrl+C` 退出，避免占用设备：

```bash
cd ~/lkx/deploy
python view_realsense_depth.py
python read_tactile_contacts.py --verbose
```

第一次先运行只读关节检查。该模式不会发送 `LowCmd`：

```bash
python deploy_student.py eth0 --joint-debug
```

机器人架空并准备好物理急停后运行：

```bash
python deploy_student.py eth0
```

程序会自动检查相机和触觉，再通过 Unitree SDK 关闭自带 sport motion mode。运行时：

- 左摇杆控制前后和横向目标。
- 按一次 `Select` 进入 STOP，再按一次恢复。
- STOP 和翻倒保护会持续发送全零命令。
- `Ctrl+C` 退出时会发送零命令并保存日志。

## 6. 参数

```text
python deploy_student.py [mujoco|网卡] [参数]
```

| 参数 | 功能 |
| --- | --- |
| `mujoco` 或不填写 | MuJoCo 模式，使用 `lo`，不会控制真机。 |
| `eth0` | Go2 真机模式。 |
| `--joint-debug` | 只读打印关节角，不发送命令。 |
| `--camera-debug` | 显示送入 CNN 的深度图。 |
| `--obs-debug` | 打印观测、estimated 和 reconstructed 信息，开销较大。 |
| `--record` | 每 0.5 s 记录一次 NPZ，退出时保存。 |
| `--record-contact` | 约 50 Hz 实时写 CSV，断电后仍能保留大部分数据。 |

常用配置在 `deploy_student_config.py`：

```text
control_dt = 0.02                       # 50 Hz
tensorrt_post_inference_delay_s = 0.008 # 前向开始到 LowCmd 写出
kp = 25.0
kd = 0.5
action_scale = 1
auto_stop_after_s = -1                  # <= 0 表示关闭
```

## 7. 日志

终端主要输出：

- `[policy]`：前向时间、策略频率、平均/最大耗时。
- `[lowcmd]`：LowCmd 写入次数、频率和间隔，正常约为 `50 Hz`、`20 ms`。
- `[joystick]`：摇杆、goal 和按键状态。
- `[student]`：运行状态、STOP 和安全信息。
- `[depth]` / `[tactile]`：传感器状态。

日志保存在 `~/lkx/deploy/records`。

```bash
python read_records.py            # 查看 --record 的 NPZ
python view_contact_records.py    # 查看 --record-contact 的 CSV
```

同步回笔记本：

```bash
cd ~/unitree_mujoco
rsync -av unitree@192.168.123.18:~/lkx/deploy/records/ deploy/records/
```
