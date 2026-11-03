# BPX HOLD：无损请求传输（2026-11-03）

## 为什么本次不放宽保护

现场连续状态显示：sensors.ready持续为true、控制约30Hz、硬限位裁剪和动作拒绝均为0，
HOLD原因是等待首块或下一动作块；每次请求含约2.07MB的原始图像。
服务器推理约90ms，而端到端存在秒级长尾。应优先降低传输负担，不能把动作等待
误判为电机毛刺门槛过严。该证据定位到传输/等待残差，不能单独区别丢包、拥塞、
OS调度、缓冲排队或准确的单向网络延迟。

## 协议与不变量

- 新服务器在metadata中声明 `request_compression: zlib-v1`。
- 新工控机将原MessagePack请求用Python标准库zlib level=1压缩。
  仅当压缩包加5字节头比原包小才采用，否则发送原包。
- 压缩帧格式为ASCII `BPXZ1` 加zlib流；回复和metadata仍是原有MessagePack。
- 服务器最多解压64MiB，拒绝损坏、截断、尾随数据和超限压缩流。
- 旧工控机→新服务器仍可直接发送原请求；新工控机→旧服务器因无能力声明而不压缩。
- 不改变任何像素、图像编码结果、分辨率、state、prompt、动作或训练函数。
  不是JPEG、不改变摄像头采集参数、不修改帧率或动作块长度。
- 压缩仅作用于本项目WebSocket请求；不改系统代理、网络路由或全局环境。

实际压缩比依画面而定，需观察工控机CPU压缩开销与传输残差，不能保证仅靠压缩
解决故障网络。持续秒级抖动仍需检查有线/无线链路、相机/遥操视频流竞争及丢包。

## 更新与验证

先人工停用策略并确认机器人妥善支撑，再更新并重启两端：

- 服务器拉取本分支，按原命令重启原模型；仍等待BPX WARMUP READY及监听。
- 工控机复制新 `scripts/openpi_websocket_client.py`、`scripts/bpx_policy_bridge.py`；
  若同时更新前次回位/HOLD修复，也复制控制台 `web/app.js`、`web/index.html`。
- 不覆盖工控机config、token、启动器或硬件驱动；只重启策略桥并强刷网页，
  不全部重启机器人、不同时启动多个策略桥。

服务器前3次、每30次或单次模型推理超过500ms时记录：
`Inference request=... wire=...B raw=...B decode=...ms infer=...ms`。
这里decode包括无损解压和MessagePack解包，infer包含正常Policy整次前向。
wire是应用层消息大小，不含TCP/WebSocket头。

工控机status的timing增加request_uncompressed_bytes、request_compression_ms；
request_bytes变为实际发送消息大小。UI请求/响应大小处悬停可查看原大小与压缩耗时。
比较更新前后端到端p95、transport_p95_ms、inference_gap_hold_count，
不以单次低时延证明连续性或以网络残差对半估计证明真实上下行延迟。

标准库服务解码测试（不导入OpenPI运行时、不加载模型）：

```bash
python3 src/openpi/serving/lossless_transport_test.py -v
```

已在开发机通过8项纯标准库实际编解码函数测试、静态语法检查。
数组/MessagePack逐像素回归规格另存于工控机tests/test_openpi_protocol.py，
遵照本机不运行项目ML/ROS环境的约定未执行该SDK测试；实际压缩比、ARM开销和
网络改善尚待用户验收，不宣称已经解决所有实机HOLD。
