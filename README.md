# KDCJC

**Key-Driven Coherent Jigsaw Cipher** — 密钥驱动的协同拼图密码

单 PNG 图像加密与还原：分块旋转打乱，还原信息隐写在像素 LSB 中，密钥由图像 MSB 派生，无需输入密码。


## 使用

```bash
pip install -r requirements.txt
python gui.py
```

Windows 可打包为 exe：`build_exe.bat` → `dist/KDCJC.exe`

## 功能

- 分块、旋转、全局打乱，可选块内平滑
- 元数据 AES 加密后分散写入全图 LSB
- GUI 支持加密、还原、进度条与保存还原图

## 注意

- 请保存为 **PNG**，避免微信/JPEG 重压缩破坏 LSB
- 开启块内平滑时还原为近似图；关闭时可无损还原
