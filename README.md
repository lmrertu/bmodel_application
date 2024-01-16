# 1.OCR-BCTR_application
## A OCR-BCTR application using sophon tpu-mlir  
我们基于OCR-BCTR系列模型的CRNN模型，使用jupyter notebook搭建一个OCR-BCR文字识别的应用。主要分为四个步骤：数据读取与可视化、数据的预处理、数据推理以及数据验证。  
1.数据读取与可视化：此文本图片的数据集来源于[Text Render](https://github.com/Sanster/text_renderer) ，使用opencv工具库的cv2.imread()函数将图片读取进来，将其可视化，并转成(1,3,32,256)尺寸。  
2.数据预处理：先将图片像素归一化到0到1范围内，再使用mean=0.5，scale=2.0的预处理将像素值正则化到-1到1范围内。  
3.数据推理：使用SophonSDK工具包中sophon-sail接口库的函数对bmodel模型进行推理，其中，sail.Engine用于创建实例并加载bmodel，engine.process用于对输入数据进行推理。  
4.数据验证：将输出的数据转化为文字形式，并展示出来。  
# 2.EDSR application
## A EDSR application using sophon tpu-mlir
我们基于EDSR模型建立一个对图片进行超分辨率的应用，经过我们的应用可以按照自己的需求将输入图片超分辨率2倍、3倍、4倍。  
在EDSR.ipynb应用中，我们将输入图片进行了3倍超分辨率，图片分辨率由（452，680）放大到（1356，2040）。  
首先将要超分辨率的图片地址输入到input_path，输入图片经过分块等预处理输入到推理模型中进行推理，再将推理得到的分块图片进行合成从而得到最终的超分辨率图片。  
如果有bm1684x板卡，可以使用板卡对推理过程进行加速。首先先将训练好的pth文件转化成onnx文件，再将onnx文件量化成f32、f16、int8形式的bmodel文件，使用板卡所带推理函数进行加速推理。
