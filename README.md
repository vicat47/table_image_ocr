# 表格形状图片 `OCR`
本项目用于对表格中的图片进行识别，并基于百度OCR API 实现。百度 `ocr` 提供每月 1000 次的免费识别额度。

依赖 `caffeine` 提供的精简 `web` 服务，可以快速搭建一个读取 `RESTful` 接口，在向 `caffeine` 存储后，只需携带 `id` 进行获取即可。关于 `caffeine` 的介绍请见 [caffeine - minimum viable backend](https://github.com/rehacktive/caffeine)

# 使用方法
1. 将 `config.example.ini` 更名为 `config.ini` 并修改其中的参数
2. 调用 `read_image` 方法进行识别与拆分

# 执行流程
```mermaid
flowchart TD
	START((开始)) --> A[读取配置文件]
	A --> B[读取图片文件]
	B --> C[查找图中的表格线框]
	C --> D[计算表格横竖线的焦点]
	D --> E[拆分表格为单元格]
	E --> F[拼接该单元格内的所有字符到同一行中]
	F --> G[发送百度ocr识图]
	G --> H[构造二维数组表格]
	H --> END((结束))
```

计算单元格内行高

- 通过 `start` 和 `end` 两个指针

```mermaid
flowchart TD
	S((初始化指针)) --> A{当前行有无像素}
	A --无--> B{end?}
	B --有--> D[将这个封闭行加入列表]
	B --无--> G
	D --> E[更新最大行高为最大值]
	E --> F[end 清空]
	F --> G[更新 start 的值为当前行]
	G --> A
	A --有--> C[更新 end 为 index]
	C --> A
```

# 最小行高

关于最小行高的处理，防止同一个字切到多行

![img](./处理最小行高.drawio.svg)